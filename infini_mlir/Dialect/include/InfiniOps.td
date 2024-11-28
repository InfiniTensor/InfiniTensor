#ifndef INFINI_OPS_TD
#define INFINI_OPS_TD

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def InfiniDialect : Dialect {
    let name = "infini";
    let summary = "Infini dialect for Deep Learning Operations";
    let cppNamespace = "infini::infinimlir";
}

def Infini_AddOp : Op<InfiniDialect, "add", [Pure]> {
  let summary = "Add operation";
  let description = [{
    "add input two tensor."
  }];
  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$input2
  );
  let results = (outs AnyRankedTensor:$output);
}


#endif // INFINI_OPS_TD