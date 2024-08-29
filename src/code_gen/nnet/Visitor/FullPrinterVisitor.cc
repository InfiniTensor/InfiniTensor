#include "code_gen/nnet/Visitor/FullPrinterVisitor.h"

namespace nnet {

void FullPrinterVisitor::visit_(const Tensor &c) {
    q.emplace_back(c->getName(), c->getSource(), c);
}

string FullPrinterVisitor::print(const Expr &root) {
    q.clear();
    std::ostringstream oss;
    dispatch(root);
    oss << "==> ROOT\n" << root->toReadable() << "\n";
    for (size_t i = 0; i < q.size(); ++i) {
        const auto &[name, routine, tensor] = q[i];
        oss << "==> " << name << " : ";
        if (routine) {
            oss << routine->toReadable() << "\n";
            if (routine->getExpr()) {
                oss << routine->getExpr()->toReadable() << "\n";
            } else
                oss << "[INFO] Source is nullptr \n";
            if (!routine->getInputs().empty()) {
                for (const auto &tensor : routine->getInputs())
                    q.emplace_back(tensor->getName(), tensor->getSource(),
                                   tensor);
            } else if (routine->getExpr())
                dispatch(routine->getExpr());
        } else
            oss << "Input Tensor " << tensor->toOutputShape() << "\n";
    }
    return oss.str();
}

const vector<tuple<string, Routine, Tensor>> &
FullPrinterVisitor::traverse(const Expr &root) {
    q.clear();
    dispatch(root);
    for (size_t i = 0; i < q.size(); ++i) {
        const auto &[name, routine, tensor] = q[i];
        if (routine) {
            // Matmul after DLT do not modify expression, so inputs has a higher
            // priority. Some OPs such as DLT have not implement source. Then
            // use inputs
            if (!routine->getInputs().empty()) {
                for (const auto &tensor : routine->getInputs())
                    dispatch(tensor);
            } else if (routine->getExpr()) {
                dispatch(routine->getExpr());
            } else {
                assert(false);
            }
        }
    }
    return q;
}

} // namespace nnet