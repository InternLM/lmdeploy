

#pragma once

namespace turbomind::core {

enum class ExchOp {
    kAdd,     //  Se ->  Rc
    kSetup,   //  Rc -> (B  -> D)
    kFetch,   // (D  ->  B)
    kUpdate,  //  B  ->  Rc
    kDel,     //  Rc ->  Se

    kPrepare,  //  D  ->  St
    kForward,
    kUnprep,  //  St ->  D
};

using BatchOp = ExchOp;

}  // namespace turbomind::core