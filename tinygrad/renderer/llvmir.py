from typing import cast
import math, struct
from tinygrad.renderer import Renderer, get_amx_tensor_cores_if_available
from tinygrad.ops import UOp, PatternMatcher, UPat, Ops, GroupOp
from tinygrad.dtype import dtypes, DType, PtrDType, truncate
from tinygrad.helpers import dedup

def ldt(dt:DType):
  if dt.vcount > 1: return f"<{dt.vcount} x {ldt(dt.scalar())}>"
  if isinstance(dt, PtrDType): return ldt(dt.base) + "*"
  return {dtypes.int8: "i8", dtypes.int16: "i16", dtypes.int32: "i32", dtypes.int64: "i64",
          dtypes.uint8: "i8", dtypes.uint16: "i16", dtypes.uint32: "i32", dtypes.uint64: "i64",
          dtypes.float16: "half", dtypes.float32: "float", dtypes.float64: "double", dtypes.bool: "i1", dtypes.void: "void"}[dt]

def lconst(x, dtype:DType):
  if dtype in dtypes.floats:
    if math.isinf(x) or math.isnan(x): return "0x%02X%02X%02X%02X%02X%02X%02X%02X" % tuple(struct.pack("d",x)[::-1])
    return truncate[dtype](x)
  return int(x)

def lcast(input_type:DType, output_type:DType):
  if dtypes.is_float(input_type):
    if dtypes.is_float(output_type): return 'fpext' if output_type.itemsize > input_type.itemsize else 'fptrunc'
    if dtypes.is_int(output_type): return 'fptoui' if dtypes.is_unsigned(output_type) else 'fptosi'
  if dtypes.is_unsigned(input_type) or dtypes.is_bool(input_type):
    if dtypes.is_float(output_type): return 'uitofp'
    if dtypes.is_int(output_type): return 'trunc' if output_type.itemsize < input_type.itemsize else 'zext'
  if dtypes.is_int(input_type):
    if dtypes.is_float(output_type): return 'sitofp'
    if dtypes.is_int(output_type): return 'trunc' if output_type.itemsize < input_type.itemsize else 'sext'
  raise NotImplementedError(f"cast from {input_type} -> {output_type} not implemented")

# llvm ops, lop[<dtype>][<op>]
unsigned_lop = { Ops.ADD: "add", Ops.MUL: "mul", Ops.IDIV: "udiv", Ops.MOD: "urem",
                 Ops.CMPLT: "icmp ult", Ops.CMPNE: "icmp ne", Ops.OR: "or", Ops.AND: "and", Ops.XOR: "xor", }
signed_lop = {**unsigned_lop, Ops.CMPLT: "icmp slt", Ops.IDIV: "sdiv", Ops.MOD: "srem"}
flags = " nsz arcp contract afn"
float_lop = {Ops.ADD: "fadd"+flags, Ops.MUL: "fmul"+flags, Ops.CMPLT: f"fcmp{flags} ult", Ops.CMPNE: f"fcmp{flags} une", Ops.FDIV: "fdiv"+flags}
lop = {**{x:unsigned_lop for x in (dtypes.bool,)+dtypes.uints}, **{x:signed_lop for x in dtypes.sints}, **{x:float_lop for x in dtypes.floats}}

llvm_rewrite = PatternMatcher([
  # memory load/store
  (UPat(Ops.INDEX, name="x"), lambda ctx,x:
   f"  {ctx[x]} = getelementptr inbounds {ldt(x.dtype.base)}, {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ldt(x.src[1].dtype)} {ctx[x.src[1]]}"),
  (UPat(Ops.LOAD, src=(UPat.var('idx'), UPat.var('alt'), UPat.var('mask')), name="x"), lambda ctx,x,idx,alt,mask:
   f"  br label {ctx[x]}_entry\n{ctx[x][1:]}_entry:\n"
   f"  br i1 {ctx[mask]}, label {ctx[x]}_load, label {ctx[x]}_exit\n{ctx[x][1:]}_load:\n"
   f"  {ctx[x]}_yes = load {ldt(x.dtype)}, {ldt(idx.dtype)} {ctx[idx]}\n"
   f"  br label {ctx[x]}_exit\n{ctx[x][1:]}_exit:\n"
   f"  {ctx[x]} = phi {ldt(x.dtype)} [{ctx[x]}_yes, {ctx[x]}_load], [{ctx[alt]}, {ctx[x]}_entry]"),
  (UPat(Ops.LOAD, src=(UPat.var('idx'),), name="x"), lambda ctx,x,idx: f"  {ctx[x]} = load {ldt(x.dtype)}, {ldt(idx.dtype)} {ctx[idx]}, align {x.src[0].dtype.itemsize}"),
  (UPat(Ops.STORE, name="x"), lambda ctx,x: f"  store {ldt(x.src[1].dtype)} {ctx[x.src[1]]}, {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, align {x.src[0].dtype.itemsize}"),

  # GEP/VECTORIZE/CAST for float4 support
  (UPat(Ops.GEP, name="x"), lambda ctx,x: f"  {ctx[x]} = extractelement {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, i32 {x.arg[0]}"),
  (UPat(Ops.VECTORIZE, src=UPat.var('y'), name="x"), lambda ctx,x,y:
   f"  {ctx[x]}_z = insertelement <1 x {ldt(y.dtype)}> poison, {ldt(y.dtype)} {ctx[y]}, i32 0\n"
   f"  {ctx[x]} = shufflevector <1 x {ldt(y.dtype)}> {ctx[x]}_z, <1 x {ldt(y.dtype)}> poison, <{x.dtype.count} x i32> zeroinitializer"),
  (UPat(Ops.VECTORIZE, name="x"), lambda ctx,x: "\n".join([(f"  {ctx[x]}_{i}" if i+1 != len(x.src) else f"  {ctx[x]}")+
                                                            f" = insertelement {ldt(x.dtype)} "+(f"{ctx[x]}_{i-1}" if i != 0 else "poison")+
                                                            f", {ldt(u.dtype)} {ctx[u]}, i32 {i}" for i,u in enumerate(x.src)])),
  (UPat(Ops.CAST, name="x"), lambda ctx,x:
   f"  {ctx[x]} = bitcast {ldt(x.src[0].dtype)} {ctx[x.src[0]]} to {ldt(x.dtype)}" if isinstance(x.dtype, PtrDType) else None),

  # unary/binary/ternary ops
  (UPat(Ops.SQRT, name="x"), lambda ctx,x:
   f"  {ctx[x]} = call{flags} {ldt(x.dtype)} @llvm.sqrt.{ldt(x.src[0].dtype)}({ldt(x.src[0].dtype)} {ctx[x.src[0]]})"),
  (UPat(Ops.BITCAST, name="x"), lambda ctx,x: f"  {ctx[x]} = bitcast {ldt(x.src[0].dtype)} {ctx[x.src[0]]} to {ldt(x.dtype)}"),
  (UPat(Ops.CAST, name="x"), lambda ctx,x: f"  {ctx[x]} = {lcast(x.src[0].dtype, x.dtype)} {ldt(x.src[0].dtype)} {ctx[x.src[0]]} to {ldt(x.dtype)}"),
  (UPat(GroupOp.Binary, name="x"), lambda ctx,x:
   f"  {ctx[x]} = {lop[x.src[0].dtype.scalar()][x.op]} {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ctx[x.src[1]]}"),
  (UPat(Ops.WHERE, name="x"), lambda ctx,x:
   f"  {ctx[x]} = select {ldt(x.src[0].dtype)} {ctx[x.src[0]]}, {ldt(x.src[1].dtype)} {ctx[x.src[1]]}, {ldt(x.src[2].dtype)} {ctx[x.src[2]]}"),

  # range
  (UPat(Ops.RANGE, name="x"), lambda ctx,x:
   f"  br label %loop_entry_{x.arg}\nloop_entry_{x.arg}:\n"
   f"  br label %loop_body_{x.arg}\nloop_body_{x.arg}:\n"
   f"  {ctx[x]} = phi {ldt(x.dtype)} [{ctx[x.src[0]]}, %loop_entry_{x.arg}], [{ctx[x]}phi, %loop_latch_{x.arg}]"),
  (UPat(Ops.ENDRANGE, name="x"), lambda ctx,x:
   f"  br label %loop_latch_{x.src[0].arg}\nloop_latch_{x.src[0].arg}:\n"
   f"  {ctx[x.src[0]]}phi = add i32 {ctx[x.src[0]]}, 1\n  {ctx[x]} = icmp ult i32 {ctx[x.src[0]]}phi, {ctx[x.src[0].src[1]]}\n"
   f"  br i1 {ctx[x]}, label %loop_body_{x.src[0].arg}, label %loop_exit_{x.src[0].arg}\nloop_exit_{x.src[0].arg}:"),

  # if
  (UPat(Ops.IF, name="x"), lambda ctx,x: f"  br i1 {ctx[x.src[0]]}, label %ifbody_{ctx[x][1:]}, label %ifskip_{ctx[x][1:]}\nifbody_{ctx[x][1:]}:"),
  (UPat(Ops.ENDIF, name="x"), lambda ctx,x: f"  br label %ifskip_{ctx[x.src[0]][1:]}\nifskip_{ctx[x.src[0]][1:]}:"),

  # wmma
  (UPat(Ops.WMMA, name="x"), lambda ctx, x: f"  {ctx[x]} = bitcast <256 x float> {ctx[x.src[2]]} to <256 x float>"),
  (UPat(Ops.WMMA, name="x"), lambda ctx, x: f"  {ctx[x]} = call {ldt(x.dtype)} @{x.arg[0]}({', '.join([f'{ldt(y.dtype)} {ctx[y]}' for y in x.src])})")
])

def llvm_bf16_cast(buf:UOp, idx:UOp, root:UOp):
  u16_buf = buf.replace(dtype=dtypes.ushort.ptr(size=cast(PtrDType,buf.dtype).size))
  return UOp.load(UOp.index(u16_buf, idx), dtype=dtypes.ushort).cast(dtypes.uint).mul(1<<16).bitcast(dtypes.float32).cast(root.dtype)

class LLVMRenderer(Renderer):
  device = "LLVM"
  supports_float4 = True
  has_local = False
  has_shared = False
  global_max = None
  tensor_cores = get_amx_tensor_cores_if_available()

  extra_matcher = PatternMatcher([
    # rewrite RECIP with FDIV
    (UPat(Ops.RECIP, name="x"), lambda x: UOp(Ops.FDIV, x.dtype, (x.const_like(1), x.src[0]))),
    # rewrite cast to bool to CMPNE 0
    (UPat(Ops.CAST, dtype=dtypes.bool, name="x"), lambda x: x.src[0] != x.src[0].const_like(0)),
    # rewrite MAX to CMPLT + WHERE
    (UPat(Ops.MAX, name="m"), lambda m: (m.src[0] < m.src[1]).where(m.src[1], m.src[0])),
    # rewrite bf16 CAST(LOAD) to CAST(BITCAST)
    (UPat(Ops.CAST, name="root", src=(UPat.load(UPat.index(UPat.var("buf"), UPat.var("idx")), dtype=dtypes.bfloat16),)), llvm_bf16_cast),
  ])

  def __init__(self, abi:str|None=None):
    self.abi = abi

  def render(self, name: str, uops: list[UOp]) -> str:
    r: dict[UOp, str] = {}
    args: list[str] = []
    prefix: list[str] = []
    kernel: list[str] = []
    end_lines: dict[str, None] = {}
    vc = -1

    # prealloc all assigns
    acc_to_assign: dict[UOp, UOp] = {}
    for u in uops:
      if u.op is Ops.ASSIGN:
        vc += 1
        r[u] = r[u.src[1]] = f"%assign{vc}"
        assert u.src[0] not in acc_to_assign, "can't assign to DEFINE_ACC twice"
        acc_to_assign[u.src[0]] = u.src[1]

    # for wmma_name, (N, M, _), dtype_in, dtype_out, _, _, _, _ in dedup([uop.arg for uop in uops if uop.op is Ops.WMMA]):
# static float256 __WMMA_16_16_1_float_float(float16 data1, float16 data2, float256 data0){
# AMX_SET(0);
# for(int ridx0 = 0; ridx0 < 16; ridx0++){ AMX(4, (int *)(&data0), (ridx0*4ull)<<56 | ridx0*64ull); }
# AMX(0, (int *)(&data2), 0ull); AMX(1, (int *)(&data1), 0ull); AMX(12, 0, 0ull);
# for(int ridx0 = 0; ridx0 < 16; ridx0++){ AMX(5, (int *)(&data0), (ridx0*4ull)<<56 | ridx0*64ull); }
# AMX_SET(1);
# return data0;

  # %4 = alloca <16 x float>, align 64
  # %5 = alloca <16 x float>, align 64
  # %6 = alloca <256 x float>, align 1024
  # %7 = ptrtoint ptr %6 to i64
  # %8 = add i64 %7, 288230376151711808
  # %9 = add i64 %7, 576460752303423616
  # %10 = add i64 %7, 864691128455135424
  # %11 = add i64 %7, 1152921504606847232
  # %12 = add i64 %7, 1441151880758559040
  # %13 = add i64 %7, 1729382256910270848
  # %14 = add i64 %7, 2017612633061982656
  # %15 = add i64 %7, 2305843009213694464
  # %16 = add i64 %7, 2594073385365406272
  # %17 = add i64 %7, 2882303761517118080
  # %18 = add i64 %7, 3170534137668829888
  # %19 = add i64 %7, 3458764513820541696
  # %20 = add i64 %7, 3746994889972253504
  # %21 = add i64 %7, 4035225266123965312
  # %22 = add i64 %7, 4323455642275677120
  # %23 = ptrtoint ptr %5 to i64
  # %24 = ptrtoint ptr %4 to i64
  # br label %57

  # store <16 x float> %125, ptr %4, align 64, !tbaa !4, !noalias !9
  # store <16 x float> %109, ptr %5, align 64, !tbaa !4, !noalias !9
  # store <256 x float> %59, ptr %6, align 1024, !tbaa !4, !noalias !9
  # call void asm sideeffect "nop\0Anop\0Anop\0A.word (u0x201000 + (17 << 5) + 0)", "~{{memory}}"() # 2; AMX set
  #       """define internal void @AMX(i32 %op, i64 %gpr, i64 %btf) {\nentry:
  # %sum = add i64 %gpr, %btf
  # %shl = shl i64 %sum, 5
  # %offset = add i64 %shl, u0x201000
  # call void asm sideeffect ".word $0", "r,~{memory}" (i64 %offset)\n  ret void\n}""",
  # call void asm sideeffect "nop\0Anop\0Anop\0A.word (u0x201000 + (17 << 5) + 1)", "~{{memory}}"() # 2; AMX clr
      # def AMX(op: int, off): return f'call void asm sideeffect ".word (0x201000+($0 << 5)+0$1-((0$1>>4)*6))", "i,r,~{{memory}}"(i32 {op}, i64 %7+{off})'
#       prefix += [
#   f"define internal {(dto := ldt(dtype_out.vec(N * N)))} @{wmma_name}({(dti := ldt(dtype_in.vec(N)))} %data1, {dti} %data2, {dto} %data0){{\nentry:",
#   # f'  call void asm sideeffect "nop\0Anop\0Anop\0A.word ({0x201000 + (17 << 5) + 0})", "~{{memory}}"()',
# # *[f"  {AMX(4, i*4<<56 | i*64)})" for i in range(16)],
# #  *["  call void @AMX(i32 0, (int *)(&data2), i64 0)", "  call void @AMX(i32 1, (int *)(&data1), i64 0)", "  call void @AMX(i32 12, 0, i64 0)"],
# #  *[f"  {AMX(4, i*4<<56 | i*64)})" for i in range(16)],
#   # f'  call void asm sideeffect "nop\0Anop\0Anop\0A.word ({0x201000 + (17 << 5) + 1})", "~{{memory}}"()',
#   f"  ret {dto} %data0\n}}"]

    for u in uops:
      # hack for defining sqrt function (TODO: can we get a transcendental for this?)
      if u.op is Ops.SQRT: end_lines[f'declare {ldt(u.dtype)} @llvm.sqrt.{ldt(u.dtype)}({ldt(u.dtype)} %".1")'] = None

      if u.op in (Ops.DEFINE_GLOBAL, Ops.DEFINE_VAR):
        r[u] = f"%data{u.arg}" if u.op is Ops.DEFINE_GLOBAL else f"%{u.arg[0]}"
        args.append(f"{ldt(u.dtype)}{' noalias' if isinstance(u.dtype, PtrDType) else ''} {r[u]}")
      elif u.op is Ops.ASSIGN: pass  # assign is already handled by the first pass
      elif u.op is Ops.DEFINE_ACC: r[u] = r[u.src[0]]  # a define acc can be used and never be assigned to
      elif u.op is Ops.CONST: r[u] = lconst(u.arg, u.dtype)
      elif u.op is Ops.CAST and ldt(u.dtype) == ldt(u.src[0].dtype): r[u] = r[u.src[0]] # cast from signed to unsigned of the same size is a noop
      else:
        # if it's an assign target, it's already preallocated
        if u not in r:
          vc += 1
          r[u] = f"%v{vc}"

        # do the rendering of the llvm ir code
        if (l:=llvm_rewrite.rewrite(u, ctx=r)) is None: raise RuntimeError(f"failed to render {u.op} with {u.dtype} srcs {[x.dtype for x in u.src]}")
        kernel.append(cast(str, l))

        # generate the phi nodes for the assigns
        if u.op is Ops.RANGE:
          for x in acc_to_assign:
            if u in x.src:  # if this range is relevant for this acc
              vc += 1
              kernel.append(f"  %acc{vc} = phi {ldt(x.dtype)}" f"[{r[x]}, %loop_entry_{u.arg}], [{r[acc_to_assign[x]]}, %loop_latch_{u.arg}]")
              r[x] = f"%acc{vc}"

    # output the function. chr(10) is '\n' (python < 3.12 doesn't support backslashes in f-strings)
    return "\n".join(prefix)+f'''\n\
define{(' '+self.abi) if self.abi is not None else ''} void @{name}({','.join(args)}) #0 {{
{chr(10).join(kernel)}
  ret void
}}
{chr(10).join(end_lines.keys())}
attributes #0 = {{ nounwind "no-builtins" "no-trapping-math"="true" }}
'''
