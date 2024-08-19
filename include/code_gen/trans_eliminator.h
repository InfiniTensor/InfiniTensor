#ifndef TRANS_ELIMINATOR_H
#define TRANS_ELIMINATOR_H

#include "generator.h"

namespace tpm {
class TransEliminator {
    std::vector<std::shared_ptr<Operator>> all_ops;
    std::shared_ptr<Reciprocity> reciprocity;

  public:
    TransEliminator();

    std::shared_ptr<SubGraph> eliminate(std::shared_ptr<SubGraph> &graph);

  private:
    bool checkValid(std::shared_ptr<SubGraph> &graph);

    int doEliminate(std::shared_ptr<SubGraph> &graph,
                    std::shared_ptr<SubGraph> &eliminated);
};
} // end of namespace tpm

#endif // TRANS_ELIMINATOR_H
