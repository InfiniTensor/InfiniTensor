#include "code_gen/cmutator.h"
#include "cstdlib"
#include "dirent.h"
using namespace tpm;

std::map<uint64_t, std::vector<std::shared_ptr<SubGraph>>>
    CMutator::l1_mutation_cache;
std::map<uint64_t, std::vector<std::shared_ptr<SubGraph>>>
    CMutator::l2_mutation_cache;

CMutator::CMutator(bool prune_reciprocity)
    : equal_threshold(0.7), num_valid_tensors(0), num_total_tensors(0),
      max_depth(3), searchingGraph(new SubGraph()),
      max_num_elements(1024 * 1024 * 1024), num_reserve_ops(0), group_size(0),
      prune_reciprocity(prune_reciprocity), computingPos({}) {
    enable_eq_opt = (getenv("PET_DISABLE_EQ_OPT") == nullptr);
    enable_non_eq_opt = (getenv("PET_DISABLE_NO_NEQ_OPT") == nullptr);
    if (!enable_non_eq_opt)
        equal_threshold = 0.99;
    printf("CMutator eq/non-eq opt status: %d/%d\n", enable_eq_opt,
           enable_non_eq_opt);
    reserveTensors(10);
}

void CMutator::run(SubGraph *in_graph, std::vector<SubGraph *> &out_graphs,
                   int mdepth,
                   std::vector<std::shared_ptr<Operator>> candidate_ops,
                   float threshold) {
    // TODO: remove and make sure all the ops in the input graph are in the
    // searching list
    // out_graphs.emplace_back(new SubGraph(in_graph->getOperators()));

    // check cache
    if (checkCache(in_graph, out_graphs)) {
        return;
    }

    // automatically add candidates ops if candidate_ops list is empty.
    // candidates_ops list will note be modified when generator is in finding
    // reciprocity mode
    group_size = 0;
    auto graph_type = statGraph(in_graph);
    if (prune_reciprocity && candidate_ops.empty()) {
        switch (graph_type) {
        case Empty:
            return;
        case Conv1X1:
            addCandidateOpsForConv1x1(candidate_ops, in_graph);
            break;
        case NormalConv: {
            auto hash =
                computeHashForSingleComputeOp(in_graph->getOperators()[0]);
            if (l1_mutation_cache.find(hash) != l1_mutation_cache.end()) {
                out_graphs.clear();
                for (auto out : l1_mutation_cache[hash]) {
                    auto new_graph = new SubGraph(out->getOperators());
                    markTransType(in_graph, new_graph);
                    if (validDepth(new_graph))
                        out_graphs.emplace_back(new_graph);
                }
                return;
            }
            addCandidateOpsForNormalConv(candidate_ops, in_graph);
            break;
        }
        case NormalOddConv:
            addCandidateOpsForNormalOddConv(candidate_ops, in_graph);
            break;
        case DilatedConv:
            addCandidateOpsForDilatedConv(candidate_ops, in_graph);
            break;
        case TransKernelConv:
            addCandidateOpsForTransKernelConv(candidate_ops, in_graph);
            break;
        case GroupConv:
            addCandidateOpsForGroupConv(candidate_ops, in_graph);
            break;
        case TransposeGroupConv:
            addCandidateOpsForTransposeGroupConv(candidate_ops, in_graph);
            break;
        case NormalMatmul:
            addCandidateOpsForNormalMatmul(candidate_ops, in_graph);
            break;
        case BatchMatmul:
            addCandidateOpsForBatchMatmul(candidate_ops, in_graph);
            break;
        default:
            return;
        }
    } else
        max_depth = mdepth > 0 ? mdepth : 3;
    equal_threshold = threshold;

    // refresh candidate op list
    for (auto &opv : all_ops)
        for (auto op : opv)
            if (op != nullptr)
                delete op;
    all_ops = std::vector<OpVec>(max_depth, OpVec());
    for (auto &opv : all_ops)
        for (auto op : candidate_ops)
            opv.emplace_back(op->clone());

    // search reciprocities for pruning
    if (prune_reciprocity)
        reciprocity = std::make_shared<Reciprocity>(candidate_ops);

    for (auto input : in_graph->getInputs())
        input->dataRand();
    computingPos.clear();
    auto outputs = in_graph->getOutputs();
    for (size_t i = 0, iEnd = outputs.size(); i < iEnd; ++i) {
        computingPos.emplace_back(std::vector<std::pair<Dim, VType>>());
        auto &back = computingPos.back();
        auto dm = outputs[i]->getDims();
        srand(time(NULL));
        for (int j = 0; j < 8; ++j) {
            Dim randPos = {};
            for (auto d : dm)
                randPos.emplace_back(((rand() % 2) + 1) * d / 3);
            back.emplace_back(std::make_pair(randPos, 0));
        }
        for (auto &pos : back) {
            auto comp = in_graph->compute(pos.first, i);
            if (!comp.first)
                return;
            pos.second = comp.second;
        }
    }

    // set tensors
    auto num_input_tensors = in_graph->getInputs().size();
    auto new_num_total_tensors = num_input_tensors + max_depth * 2;
    // allocate new tensors if the backup tensors are not enough
    reserveTensors(new_num_total_tensors);
    // set data and state for input tensors
    for (size_t i = 0; i < num_input_tensors; ++i) {
        auto input = in_graph->getInputs()[i];
        auto tensor = searchingGraph->getTensors()[i];
        tensor->clone(input);
        tensor->setData(input->getDataPtr());
        tensor->initSplittingPoints();
    }

    if (enable_box_verification) {
        // fully compute input graph
        for (int i = 0; i < (int)in_graph->getOutputs().size(); ++i)
            in_graph->compute(in_graph->getOutputs()[i]->getDims(), i, true);
        // initialize splitting points for both the in_graph and searching graph
        for (auto input : in_graph->getInputs())
            input->initSplittingPoints();
        // infer splitting points for in_graph
        for (auto op : in_graph->getOperators()) {
            op->inferSplittingPoints();
        }
    }

    num_valid_tensors = num_input_tensors;

    std::unordered_set<uint64_t> visited;
    switch (graph_type) {

    case Conv1X1: {
        addPreprocessForConv1x1(in_graph);
        dfs(oplist.size(), in_graph, searchingGraph, out_graphs, visited);
        resetGraph(in_graph);
        dfs(oplist.size(), in_graph, searchingGraph, out_graphs, visited);
        break;
    }

    case NormalConv: {
        splitGroupConv(in_graph, out_graphs);
        // break;
        while (!oplist.empty())
            popBackOp();
        while (num_valid_tensors > in_graph->getInputs().size())
            popBackTensor();
        dfs(oplist.size(), in_graph, searchingGraph, out_graphs, visited);
        break;
    }

    case TransKernelConv: {
        addPreprocessForTransKernel(in_graph);
        dfs(oplist.size(), in_graph, searchingGraph, out_graphs, visited);
        break;
    }

    case GroupConv: {
        addPreprocessForGroupConvGCD(in_graph);
        dfs(oplist.size(), in_graph, searchingGraph, out_graphs, visited);
        resetGraph(in_graph);
        addPreprocessForGroupConvMAX(in_graph);
        dfs(oplist.size(), in_graph, searchingGraph, out_graphs, visited);
        resetGraph(in_graph);
        addPreprocessForGroupConvOneInput(in_graph);
        SubGraph *new_graph = new SubGraph(oplist);
        auto newOutputs = new_graph->getOutputs();
        for (size_t i = 0, j = 0, iEnd = newOutputs.size(); i < iEnd; ++i) {
            if (newOutputs[i]->isNotCounted())
                continue;
            newOutputs[i]->clone(in_graph->getOutputs()[j]);
            ++j;
        }
        if (prune_reciprocity)
            markTransType(in_graph, new_graph);
        if (validDepth(new_graph)) {
            out_graphs.emplace_back(new_graph);
            for (auto sg : out_graphs)
                sg->print();
        }
        break;
    }

    case TransposeGroupConv: {
        addPreprocessForTransposeGroupConvRS(in_graph);
        dfs(oplist.size(), in_graph, searchingGraph, out_graphs, visited);
        resetGraph(in_graph);
        addPreprocessForTransposeGroupConvSR(in_graph);
        dfs(oplist.size(), in_graph, searchingGraph, out_graphs, visited);
        break;
    }

    case NormalOddConv: {
        addPreprocessForPadSlice(in_graph);
        dfs(oplist.size(), in_graph, searchingGraph, out_graphs, visited);
        break;
    }

    case BatchMatmul: {
        addPreprocessForBatchMatmul(in_graph);
        dfs(oplist.size(), in_graph, searchingGraph, out_graphs, visited);
        break;
    }

    default: {
        dfs(oplist.size(), in_graph, searchingGraph, out_graphs, visited);
        break;
    }
    }

    // update cache
    if (out_graphs.size() != 0) {
        updateCache(in_graph, out_graphs);
    }

    clearOps();
    clearTensors(0);

    num_reserve_ops = 0;
    group_size = -1;
}

void CMutator::dfs(int depth, SubGraph *in_graph, SubGraph *cur_graph,
                   std::vector<SubGraph *> &out_graphs,
                   std::unordered_set<uint64_t> &visited) {
    // Connect the graph
    if (!cur_graph->resetOps(oplist, num_valid_tensors)) {
        return;
    }

    // Prune if having searched an identical graph
    if (!visited.insert(cur_graph->getHash()).second) {
        return;
    }

    // prune reciprocities
    if (prune_reciprocity && reciprocity->is_tail_reciprocity(oplist))
        return;

    // If full_computing is true, there is no conv/gemm ops and all ops are
    // computed instead of sampling verification
    bool full_computing = true;
    for (auto op : oplist) {
        if (!op->isTransposeOp()) {
            full_computing = false;
            break;
        }
    }

    if (oplist.size() > num_reserve_ops) {
        // legal mutant is found, finish the DFS
        // Only non-computing op will not be a mutant
        // TODO: when dim compute for transpose is ready
        if (!full_computing || !prune_reciprocity) {
            if (is_a_mutant(cur_graph, in_graph, full_computing)) {
                // // debug
                // for (auto candidate : candidates) {
                //     candidate->print();
                //     for (auto i : candidate->getTensors()) {
                //         printf("tensor guid: %lu ", i->getGuid());
                //         i->printSplittingPoints();
                //         printf("\n");
                //     }
                // }
                // Copy the graph by value
                SubGraph *new_graph = new SubGraph(oplist);
                auto newOutputs = new_graph->getOutputs();
                for (size_t i = 0, j = 0, iEnd = newOutputs.size(); i < iEnd;
                     ++i) {
                    if (newOutputs[i]->isNotCounted())
                        continue;
                    newOutputs[i]->clone(in_graph->getOutputs()[j]);
                    ++j;
                }
                if (prune_reciprocity)
                    markTransType(in_graph, new_graph);
                if (validDepth(new_graph))
                    out_graphs.emplace_back(new_graph);
                return;
            }
        }
    }

    if (depth >= max_depth)
        return;

    for (size_t index = 0; index < all_ops[depth].size(); index++) {
        Operator *op = all_ops[depth][index];
        assert(op->isClear());
        if (op->getType() == Operator::Split) {
            // assert(op->getInputs().size() == 1);
            SplitOp *split = (SplitOp *)op;
            for (size_t i = 0; i < num_valid_tensors; i++) {
                Tensor *x = cur_graph->getTensors()[i];
                TensorVec outs;
                auto outsNum = split->getSizes().size();
                for (size_t j = 0; j < outsNum; ++j)
                    outs.emplace_back(newTensor());
                // Tensor *o1 = newTensor(); // memory allocator
                // Tensor *o2 = newTensor(); // memory allocator
                // TODO: mark tensor as full computed after full computing
                // if ((full_computing && !split->compute({x}, {o1, o2})) ||
                //     (!full_computing && !split->computeShape({x}, {o1, o2})))
                //     { popBackTensor(op); popBackTensor(op); continue;
                // }
                if ((full_computing && !split->compute({x}, outs)) ||
                    (!full_computing && !split->computeShape({x}, outs))) {
                    for (size_t j = 0; j < outsNum; ++j)
                        popBackTensor(op);
                    // popBackTensor(op);
                    // popBackTensor(op);
                    continue;
                }
                // prune if a same op with the same inputs exists
                if (have_same_op(op)) {
                    for (size_t j = 0; j < outsNum; ++j)
                        popBackTensor(op);
                    // popBackTensor(op);
                    // popBackTensor(op);
                    continue;
                }
                pushBackOp(op);
                dfs(depth + 1, in_graph, cur_graph, out_graphs, visited);
                popBackOp();
                for (size_t j = 0; j < outsNum; ++j)
                    popBackTensor();
                // popBackTensor();
                // popBackTensor();
            }
        } else if (op->getType() == Operator::Concat && group_size > 1) {
            TensorVec inputTensors, weightTensors;
            for (size_t i = group_size * 2; i < num_valid_tensors; ++i) {
                auto t = searchingGraph->getTensors()[i];
                if (t->getType() == Tensor::Input)
                    inputTensors.emplace_back(t);
            }
            for (size_t i = 0; i < num_valid_tensors; ++i) {
                auto t = searchingGraph->getTensors()[i];
                if (t->getType() == Tensor::Weight)
                    weightTensors.emplace_back(t);
            }
            if (inputTensors.size() == group_size) {
                auto out = newTensor();
                if ((full_computing && !op->compute(inputTensors, {out})) ||
                    (!full_computing &&
                     !op->computeShape(inputTensors, {out}))) {
                    popBackTensor(op);
                    // popBackTensor(op);
                    // popBackTensor(op);
                    continue;
                }
                // prune if a same op with the same inputs exists
                if (have_same_op(op)) {
                    popBackTensor(op);
                    // popBackTensor(op);
                    // popBackTensor(op);
                    continue;
                }
                pushBackOp(op);
                dfs(depth + 1, in_graph, cur_graph, out_graphs, visited);
                popBackOp();
            } else if (inputTensors.size() == 1) {
                auto out = newTensor();
                for (size_t i = 1; i < group_size; ++i)
                    inputTensors.emplace_back(inputTensors[0]);
                if ((full_computing && !op->compute(inputTensors, {out})) ||
                    (!full_computing &&
                     !op->computeShape(inputTensors, {out}))) {
                    popBackTensor(op);
                    // popBackTensor(op);
                    // popBackTensor(op);
                    continue;
                }
                // prune if a same op with the same inputs exists
                if (have_same_op(op)) {
                    popBackTensor(op);
                    // popBackTensor(op);
                    // popBackTensor(op);
                    continue;
                }
                pushBackOp(op);
                dfs(depth + 1, in_graph, cur_graph, out_graphs, visited);
                popBackOp();
            }
            if (weightTensors.size() == group_size) {
                auto out = newTensor();
                if ((full_computing && !op->compute(weightTensors, {out})) ||
                    (!full_computing &&
                     !op->computeShape(weightTensors, {out}))) {
                    popBackTensor(op);
                    // popBackTensor(op);
                    // popBackTensor(op);
                    continue;
                }
                // prune if a same op with the same inputs exists
                if (have_same_op(op)) {
                    popBackTensor(op);
                    // popBackTensor(op);
                    // popBackTensor(op);
                    continue;
                }
                pushBackOp(op);
                dfs(depth + 1, in_graph, cur_graph, out_graphs, visited);
                popBackOp();
            } else if (weightTensors.size() == 1) {
                auto out = newTensor();
                for (size_t i = 1; i < group_size; ++i)
                    weightTensors.emplace_back(weightTensors[0]);
                if ((full_computing && !op->compute(weightTensors, {out})) ||
                    (!full_computing &&
                     !op->computeShape(weightTensors, {out}))) {
                    popBackTensor(op);
                    // popBackTensor(op);
                    // popBackTensor(op);
                    continue;
                }
                // prune if a same op with the same inputs exists
                if (have_same_op(op)) {
                    popBackTensor(op);
                    // popBackTensor(op);
                    // popBackTensor(op);
                    continue;
                }
                pushBackOp(op);
                dfs(depth + 1, in_graph, cur_graph, out_graphs, visited);
                popBackOp();
            }
        } else if (op->numInputs() == 2) {
            // std::cout << "Current oplist: " << oplist.size() << " " << depth
            //           << " " << std::endl;
            // for (auto op : oplist) {
            //     op->print();
            //     std::cout << " - ";
            // }
            // std::cout << std::endl;
            // for (size_t i = 0; i < num_valid_tensors; i++) {
            for (size_t i = group_size * 3; i < num_valid_tensors; i++) {
                if (op->isComputeOp() && oplist.empty())
                    continue;
                // pruning successive ComputeOp
                if (op->isComputeOp() &&
                    have_computeOp_ancestor(cur_graph->getTensors()[i]))
                    continue;
                // for (size_t j = 0; j < num_valid_tensors; j++) {
                for (size_t j = group_size * 3; j < num_valid_tensors; j++) {
                    // pruning successive ComputeOp
                    if (op->isComputeOp() &&
                        have_computeOp_ancestor(cur_graph->getTensors()[j]))
                        continue;
                    // Inputs can be the same
                    Tensor *x = cur_graph->getTensors()[i];
                    Tensor *y = cur_graph->getTensors()[j];
                    Tensor *output = newTensor();
                    auto local_full_computing = full_computing;
                    if (op->isComputeOp()) {
                        local_full_computing = false;
                    }

                    // TODO: mark tensor as full computed after full computing
                    if ((local_full_computing &&
                         !op->compute({x, y}, {output})) ||
                        (!local_full_computing &&
                         !op->computeShape({x, y}, {output}))) {
                        popBackTensor(op);
                        continue;
                    }
                    // prune if a same op with the same inputs exists
                    if (have_same_op(op)) {
                        popBackTensor(op);
                        continue;
                    }
                    pushBackOp(op);
                    dfs(depth + 1, in_graph, cur_graph, out_graphs, visited);
                    popBackOp();
                    popBackTensor();
                }
            }
        } else if (op->numInputs() == 1) {
            // std::cout << "Current oplist: " << oplist.size() << " "
            // << depth
            // << " " << std::endl; for (auto op : oplist) {
            // op->print(); std::cout << " - ";
            //}
            // std::cout << std::endl;
            // for (size_t i = num_valid_tensors - 1; i != ~(size_t)0; i--) {
            // for (size_t i = 0; i < num_valid_tensors; i++) {
            for (size_t i = group_size * 3; i < num_valid_tensors; i++) {
                Tensor *x = cur_graph->getTensors()[i];
                Tensor *output = newTensor();
                // TODO: make sure there is no bug
                // Sub-graph with only transpose op donot need to be computed
                // auto local_full_computing =
                //     full_computing && !op->isComputeOp();
                // // auto local_full_computing = !prune_reciprocity;
                // // TODO: mark tensor as full computed after full computing
                // if ((local_full_computing && !op->compute(x, output)) ||
                //     (!local_full_computing && !op->computeShape(x, output)))
                //     { popBackTensor(op); continue;
                // }
                if (prune_reciprocity) {
                    if (!op->computeShape({x}, {output})) {
                        popBackTensor(op);
                        continue;
                    }
                } else {
                    if (!op->compute({x}, {output})) {
                        popBackTensor(op);
                        continue;
                    }
                }
                // for search reciprocity
                // TODO: nicer implementation?
                // if (!prune_reciprocity && !local_full_computing &&
                //     !op->compute(x, output)) {
                //     popBackTensor(op);
                //     continue;
                // }
                // prune if a same op with the same inputs exists
                if (have_same_op(op)) {
                    popBackTensor(op);
                    continue;
                }
                pushBackOp(op);
                dfs(depth + 1, in_graph, cur_graph, out_graphs, visited);
                popBackOp();
                popBackTensor();
            }
        }
    }
}

bool CMutator::is_a_mutant(const SubGraph *mutant_graph,
                           const SubGraph *input_graph, bool full_computing) {
    size_t mouts = 0;
    for (auto output : mutant_graph->getOutputs()) {
        if (!output->isNotCounted())
            mouts++;
    }

    // already called mutant_graph->resetOps
    if (mouts != input_graph->getOutputs().size())
        return false;
    std::vector<int> matched(mutant_graph->getOutputs().size());
    for (size_t midx = 0, iidx = 0; midx < mutant_graph->getOutputs().size();
         ++midx) {
        if (mutant_graph->getOutputs()[midx]->getType() == Tensor::NotCounted)
            continue;
        if (full_computing) {
            if (!approx_equal(mutant_graph->getOutputs()[midx],
                              input_graph->getOutputs()[iidx]))
                return false;
        } else if (!enable_box_verification) {
            // sampling points
            if (!approx_equal(mutant_graph, midx, input_graph, iidx))
                return false;
        } else {
            // printf("=============================================\n");
            // bool sampling = approx_equal(mutant_graph, idx, input_graph,
            // idx);
            bool splitting = approx_equal_splitting_points(mutant_graph, midx,
                                                           input_graph, iidx);
            // if (sampling != splitting) {
            //     printf("Inconsistent sampling=%d splitting=%d\nInput graph:
            //     ",
            //            sampling, splitting);
            //     const_cast<SubGraph *>(input_graph)->print();
            //     printf("\nMutant graph: ");
            //     const_cast<SubGraph *>(mutant_graph)->print();
            //     printf("\n");

            //     for (auto op : mutant_graph->getOperators()) {
            //         for (auto t : op->getInputs()) {
            //             printf("Tensor %lu:\n", t->getHash());
            //             printf("-- splitting points:");
            //             t->printSplittingPoints();
            //             printf("\n-- sharp:");
            //             for (int v : t->getDims())
            //                 printf(" %d,", v);
            //             printf("\n");
            //             // t->print();
            //         }
            //     }

            //     for (auto t : mutant_graph->getOutputs()) {
            //         printf("Tensor %lu:\n", t->getHash());
            //         printf("-- splitting points:");
            //         t->printSplittingPoints();
            //         printf("\n-- sharp:");
            //         for (int v : t->getDims())
            //             printf(" %d,", v);
            //         printf("\n");
            //         // t->print();
            //     }
            // }
            if (!splitting)
                return false;
        }
        ++iidx;
    }
    // for (size_t in_idx = 0; in_idx < input_graph->getOutputs().size();
    //      in_idx++) {
    //     int found = -1;
    //     // Tensor *in_tensor = input_graph->getOutputs()[in_idx];
    //     for (size_t idx = 0; idx < mutant_graph->getOutputs().size(); idx++)
    //     {
    //         assert(idx < (size_t)num_valid_tensors);
    //         if (!matched[idx] &&
    //             (full_computing
    //                  ? approx_equal(mutant_graph->getOutputs()[idx],
    //                                 input_graph->getOutputs()[in_idx])
    //                  : approx_equal(mutant_graph, idx, input_graph, in_idx)))
    //                  {
    //             found = idx;
    //             matched[idx] = true;
    //         }
    //     }
    //     if (found == -1)
    //         return false;
    //     // std::cout << "found: " << found << std::endl;
    //     // mutant_graph->getOutputs()[in_idx] =
    //     // mutant_graph->getTensors()[found];
    // }
    return true;
}

bool CMutator::approx_equal_splitting_points(const SubGraph *mutant_graph,
                                             size_t midx,
                                             const SubGraph *input_graph,
                                             size_t iidx) {
    if (mutant_graph->getOutputs()[midx]->getDims() !=
        input_graph->getOutputs()[iidx]->getDims())
        return false;
    auto ans_tensor = input_graph->getOutputs()[iidx];
    auto mut_tensor = mutant_graph->getOutputs()[midx];
    Dim dims = ans_tensor->getDims();
    // fprintf(stderr, "dims %lu:", dims.size());
    // for (auto v : dims)
    //     printf("%d ", v);
    // printf("\n");
    SplittingPoints merged(dims.size());
    // merge splitting points from two subgraphs
    auto const &ans_splitting_points = *ans_tensor->getSplittingPoints();
    auto const &mut_splitting_points = *mut_tensor->getSplittingPoints();
    for (int i = 0; i < (int)dims.size(); ++i) {
        merged[i].resize(ans_splitting_points[i].size() +
                         mut_splitting_points[i].size() + 1);
        merged[i][0] = 0;
        std::merge(ans_splitting_points[i].begin(),
                   ans_splitting_points[i].end(),
                   mut_splitting_points[i].begin(),
                   mut_splitting_points[i].end(), merged[i].begin() + 1);
        merged[i].erase(std::unique(merged[i].begin(), merged[i].end()),
                        merged[i].end());
    }

    // // debug
    // printf("ans_tensor->printSplittingPoints(): ");
    // ans_tensor->printSplittingPoints();
    // printf("\nmut_tensor->printSplittingPoints(): ");
    // mut_tensor->printSplittingPoints();
    // printf("\nmerged: ");
    // printf("[");
    // for (auto &vs : merged) {
    //     printf("[");
    //     for (auto v : vs)
    //         printf("%2d,", v);
    //     printf("],");
    // }
    // printf("]\n");

    int box_cnt = 1;
    for (const auto &one_dim : merged)
        box_cnt *= (int)one_dim.size();
    // pair<points, box_id>
    std::vector<std::pair<Dim, int>> verified_points;
    // pos is the begin position of the box. merge_index is the corresponding
    // index in the merged array.
    std::vector<int> box_element_cnt, box_error_cnt(box_cnt);

    // generate verified points for verification
    {
        std::vector<int> merged_index, pos;
        std::function<void(int, std::vector<int> &, std::vector<int> &)> dfs =
            [&dfs, &verified_points, &merged, &dims,
             &box_element_cnt](int depth, std::vector<int> &merged_index,
                               std::vector<int> &pos) {
                // pos is the
                if (depth == (int)merged.size()) {
                    int box_id = box_element_cnt.size(), element_cnt = 1;
                    for (int i = 0; i < (int)merged.size(); ++i) {
                        if (merged_index[i] == (int)merged[i].size() - 1)
                            element_cnt *=
                                (dims[i] - merged[i][merged_index[i]]);
                        else
                            element_cnt *= (merged[i][merged_index[i] + 1] -
                                            merged[i][merged_index[i]]);
                    }
                    box_element_cnt.emplace_back(element_cnt);
                    verified_points.emplace_back(std::make_pair(pos, box_id));
                    for (int i = 0; i < (int)merged.size(); ++i) {
                        if ((merged_index[i] < (int)merged[i].size() - 1 &&
                             merged[i][merged_index[i]] + 1 <
                                 merged[i][merged_index[i] + 1]) ||
                            (merged_index[i] == (int)merged[i].size() - 1 &&
                             pos[i] + 1 < (int)dims[i])) {
                            pos[i]++;
                            verified_points.emplace_back(
                                std::make_pair(pos, box_id));
                            pos[i]--;
                        }
                    }
                    return;
                }
                for (int i = 0; i < (int)merged[depth].size(); ++i) {
                    merged_index.emplace_back(i);
                    pos.emplace_back(merged[depth][i]);
                    dfs(depth + 1, merged_index, pos);
                    pos.pop_back();
                    merged_index.pop_back();
                }
            };
        dfs(0, merged_index, pos);
    }
    assert(box_cnt == (int)box_element_cnt.size());

    // // debug
    // printf("verified_points: %lu\n", verified_points.size());
    // printf("verified_points: [");
    // for (auto &vs : verified_points) {
    //     printf("[");
    //     for (auto v : vs.first)
    //         printf("%2d,", v);
    //     printf("],\n");
    // }
    // printf("]\n");

    // verify the data on the samping points
    for (auto &pos_boxId : verified_points) {
        auto comp = mutant_graph->compute(pos_boxId.first, midx);
        if (!comp.first) {
            // printf("mutant graph cannot compute: ");
            // for (auto v : pos_boxId.first)
            //     printf(" %d", v);
            // puts("");
            return false;
        }
        if (input_graph->getOutputs()[iidx]->getData(pos_boxId.first) !=
            comp.second)
            box_error_cnt[pos_boxId.second]++;

        // printf("Verification points: ");
        // for (auto p : pos_boxId.first)
        //     printf("%d,", p);
        // printf((comp.second ==
        //         input_graph->getOutputs()[output_idx]->getData(pos_boxId.first))
        //            ? ": %d %d\n"
        //            : ": %d %d XXX\n",
        //        comp.second,
        //        input_graph->getOutputs()[output_idx]->getData(pos_boxId.first));
    }

    // calculate weighted accuracy by num of elements in a box
    int equal = 0, total = 1;
    for (auto len : dims)
        total *= len;
    for (int i = 0; i < box_cnt; ++i) {
        if (box_error_cnt[i] == 0)
            equal += box_element_cnt[i];
    }
    // debug
    // printf("accuracy %lf\n", double(equal) / total);
    return (float(equal) / total > equal_threshold);
}

bool CMutator::approx_equal(const SubGraph *mutant_graph, size_t midx,
                            const SubGraph *input_graph, size_t iidx) {
    if (mutant_graph->getOutputs()[midx]->getDims() !=
        input_graph->getOutputs()[iidx]->getDims())
        return false;
    int equal = 0, total = 0;
    for (auto &pos : computingPos[iidx]) {
        auto comp = mutant_graph->compute(pos.first, midx);
        if (!comp.first)
            return false;
        if (comp.second == pos.second)
            equal++;
        total++;
    }
    // if (equal > 0) {
    //     std::cout << std::endl;
    //     std::cout << "approx_equal: " << equal << "/" << total << std::endl;
    // }
    if (float(equal) / total > equal_threshold)
        return true;
    return false;
}

bool CMutator::approx_equal(Tensor *a, Tensor *b) {
    // std::cout << "approx_equal" << std::endl;
    if (a->getDims() != b->getDims())
        return false;
    size_t equal = 0, total = 0;
    VType *a_ptr = a->getDataPtr();
    VType *b_ptr = b->getDataPtr();
    // std::cout << "Comparing tensor " << (a->getGuid()) << " with
    // tensor "
    // << (b->getGuid()) << std::endl; std::cout << "a_ptr: " << a_ptr
    // << ", b_ptr: " << b_ptr << std::endl;
    for (size_t i = 0; i < a->size(); i++) {
        // if (i % 32 == 0)
        //     std::cout << a_ptr[i] << "." << b_ptr[i] << " ";
        if (a_ptr[i] == b_ptr[i])
            equal++;
        total++;
    }
    // std::cout << std::endl;
    // std::cout << "approx_equal: " << equal << "/" << total <<
    // std::endl;
    if (float(equal) / total > equal_threshold)
        return true;
    return false;
}

bool CMutator::pushBackOp(Operator *op) {
    oplist.emplace_back(op);
    if (enable_box_verification)
        op->inferSplittingPoints();
    return true;
}

bool CMutator::popBackOp() {
    auto op = oplist.back();
    op->clear();
    oplist.pop_back();
    return true;
}

bool CMutator::clearOps() {
    while (!oplist.empty())
        popBackOp();
    return true;
}

Tensor *CMutator::newTensor() {
    if (num_valid_tensors >= searchingGraph->getTensors().size())
        reserveTensors(num_valid_tensors + 4);
    auto tensor = searchingGraph->getTensors()[num_valid_tensors];
    assert(tensor->isClear());
    num_valid_tensors++;
    return tensor;
}

void CMutator::reserveTensors(size_t size) {
    if (num_total_tensors >= size)
        return;
    Dim max_dim = {(int)max_num_elements};
    for (size_t i = num_total_tensors; i < size; ++i)
        searchingGraph->tensor(max_dim)->dataMalloc();
    num_total_tensors = size;
}

bool CMutator::popBackTensor(Operator *op) {
    num_valid_tensors--;
    searchingGraph->getTensors()[num_valid_tensors]->clear();
    if (op != nullptr)
        op->clear();
    return true;
}

bool CMutator::clearTensors(size_t threshold) {
    while (threshold < num_valid_tensors)
        popBackTensor();
    return true;
}

bool CMutator::have_computeOp_ancestor(Tensor *tensor) {
    std::function<bool(Tensor *)> backtrace =
        [&backtrace](Tensor *current) -> bool {
        auto fatherOp = current->getOutputOf();
        if (!fatherOp)
            return false;
        if (fatherOp->isComputeOp())
            return true;
        for (auto ancestorTensor : fatherOp->getInputs())
            if (backtrace(ancestorTensor))
                return true;
        return false;
    };
    return backtrace(tensor);
}

bool CMutator::have_same_op(Operator *new_op) {
    for (auto exist_op : oplist) {
        if (new_op->getHash() != exist_op->getHash())
            continue;
        if (new_op->getInputs()[0]->getHash() !=
            exist_op->getInputs()[0]->getHash())
            continue;
        if (new_op->getInputs().size() > 1 &&
            new_op->getInputs()[1]->getHash() !=
                exist_op->getInputs()[1]->getHash())
            continue;
        return true;
    }
    return false;
}

CMutator::SGType CMutator::statGraph(SubGraph *sg) {
    auto ops = sg->getOperators();
    switch (ops.size()) {
    case 0: {
        return Empty;
        break;
    }

    case 1: {
        if (ops[0]->getType() == Operator::Conv) {
            auto weight = ops[0]->getInputs()[1];
            auto r = weight->getDims()[2];
            auto s = weight->getDims()[3];
            if (((ConvOp *)sg->getOperators()[0])->getDh() == 1 &&
                ((ConvOp *)sg->getOperators()[0])->getDw() == 1 && r == 1 &&
                s == 1) {
                return Conv1X1;
            } else if (((ConvOp *)sg->getOperators()[0])->getDh() == 2 ||
                       ((ConvOp *)sg->getOperators()[0])->getDw() == 2) {
                return DilatedConv;
            } else {
                const Dim &inDim = ops[0]->getInputs()[0]->getDims();
                const Dim &wDim = ops[0]->getInputs()[1]->getDims();
                if (inDim[2] % 2 == 1 && inDim[3] % 2 == 1)
                    return NormalOddConv;
                else if (wDim[2] != wDim[3])
                    return TransKernelConv;
                else
                    return NormalConv;
            }
        } else if (ops[0]->getType() == Operator::Matmul) {
            return NormalMatmul;
        }
        break;
    }

    default:
        auto ty = ops[0]->getType();
        for (size_t i = 1, iEnd = ops.size(); i < iEnd; ++i) {
            if (ops[i]->getType() != ty)
                return Others;
        }
        if (ty == Operator::Conv) {
            std::vector<ConvOp *> convs;
            for (auto op : ops)
                convs.emplace_back(dynamic_cast<ConvOp *>(op));
            // TODO: 1x1 conv enlarge. 1x1 conv has 0 padding
            for (size_t i = 1, iEnd = ops.size(); i < iEnd; ++i)
                if (!convs[i]->same(*convs[0]))
                    return Others;
            auto inDim = ops[0]->getInputs(0)->getDims();
            // TODO: enlarge input tensor?
            for (size_t i = 1, iEnd = ops.size(); i < iEnd; ++i)
                if (ops[i]->getInputs(0)->getDims() != inDim)
                    return Others;
            auto weightDim = ops[0]->getInputs(1)->getDims();
            auto groupFlag = true;
            // TODO: kernel enlarge to group?
            for (size_t i = 1, iEnd = ops.size(); i < iEnd; ++i) {
                auto wDim = ops[i]->getInputs(1)->getDims();
                if (!(wDim[1] == weightDim[1] && wDim[2] == weightDim[2] &&
                      wDim[3] == weightDim[3] && wDim[2] == wDim[3])) {
                    groupFlag = false;
                    break;
                }
            }
            if (groupFlag)
                return GroupConv;
            auto transGroupFlag = true;
            // TODO: transpose group conv with different f dim?
            for (size_t i = 1, iEnd = ops.size(); i < iEnd; ++i) {
                auto wDim = ops[i]->getInputs(1)->getDims();
                if (!(wDim[0] == weightDim[0] && wDim[1] == weightDim[1] &&
                      ((wDim[2] == weightDim[2] && wDim[3] == weightDim[3]) ||
                       (wDim[2] == weightDim[3] && wDim[3] == weightDim[2])))) {
                    transGroupFlag = false;
                    break;
                }
            }
            if (transGroupFlag)
                return TransposeGroupConv;
        } else if (ty == Operator::Matmul) {
            // check same input shape or not
            for (int i = 0; i < (int)ops.size() - 1; ++i) {
                assert(dynamic_cast<MatmulOp *>(ops[i])->getTransA() ==
                       dynamic_cast<MatmulOp *>(ops[i + 1])->getTransA());
                assert(dynamic_cast<MatmulOp *>(ops[i])->getTransB() ==
                       dynamic_cast<MatmulOp *>(ops[i + 1])->getTransB());
                if (ops[i]->getInputs()[0]->getDims() !=
                    ops[i + 1]->getInputs()[0]->getDims()) {
                    return Others;
                }
                if (ops[i]->getInputs()[1]->getDims() !=
                    ops[i + 1]->getInputs()[1]->getDims()) {
                    return Others;
                }
            }
            return BatchMatmul;
        }
        // TODO: others?
        break;
    }

    return Others;
}

void CMutator::addCandidateOpsForConv1x1(
    std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg) {}

void CMutator::addCandidateOpsForNormalConv(
    std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg) {
    auto conv = dynamic_cast<ConvOp *>(sg->getOperators()[0]);
    // conv->print();
    // n->h
    candidate_ops.emplace_back(
        new tpm::TransposeOp(0, {0, 1, {-1, 2}, 3}, 2, TransposeOp::N2H));
    candidate_ops.emplace_back(
        new tpm::TransposeOp(2, {{0, 2}, 1, -1, 3}, -2, TransposeOp::H2N));

    // n->w
    candidate_ops.emplace_back(
        new tpm::TransposeOp(0, {0, 1, 2, {-1, 3}}, 2, TransposeOp::N2W));
    candidate_ops.emplace_back(
        new tpm::TransposeOp(3, {{0, 3}, 1, 2, -1}, -2, TransposeOp::W2N));

    // c->h
    candidate_ops.emplace_back(
        new tpm::TransposeOp(1, {0, 1, {2, -1}, 3}, 2, TransposeOp::C2H));
    candidate_ops.emplace_back(
        new tpm::ConvOp(conv->getPaddingMode(), 2, 1, 1, 1));

    // c->w
    candidate_ops.emplace_back(
        new tpm::TransposeOp(1, {0, 1, 2, {3, -1}}, 2, TransposeOp::C2W));
    candidate_ops.emplace_back(
        new tpm::ConvOp(conv->getPaddingMode(), 1, 2, 1, 1));

    // c->hw
    candidate_ops.emplace_back(
        new tpm::ConvOp(conv->getPaddingMode(), 2, 2, 1, 1));
    // candidate_ops.emplace_back(new tpm::ConvOp(4, 2, 4, 2, 1,
    // 1));
    // candidate_ops.emplace_back(new tpm::ConvOp(2, 4, 2, 4, 1,
    // 1));
    // candidate_ops.emplace_back(new tpm::ConvOp(4, 4, 4, 4, 1,
    // 1));

    // origin op
    candidate_ops.emplace_back(
        new tpm::ConvOp(conv->getPaddingMode(), 1, 1, 1, 1));
    max_depth = 3;
}

void CMutator::addCandidateOpsForNormalOddConv(
    std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg) {
    // addCandidateOpsForNormalConv(candidate_ops, sg);
    NEQOPT
    auto conv = dynamic_cast<ConvOp *>(sg->getOperators()[0]);

    // pad -> conv -> unpad
    candidate_ops.emplace_back(new tpm::PadOp({0, 0, 0, 0}, {0, 0, 1, 1}));
    candidate_ops.emplace_back(new tpm::SliceOp({0, 0, 0, 0}, {0, 0, 1, 1}));
    candidate_ops.emplace_back(
        new tpm::ConvOp(conv->getPaddingMode(), 1, 1, 1, 1));
    max_depth = 3;
}

void CMutator::addCandidateOpsForDilatedConv(
    std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg) {
    auto conv = dynamic_cast<ConvOp *>(sg->getOperators()[0]);
    // TODO: do we need to identify d2hw and hw2d?
    NEQOPT
    candidate_ops.emplace_back(
        new tpm::TransposeOp(2, {0, 1, {-1, 2}, 3}, 2, TransposeOp::D2H));
    candidate_ops.emplace_back(
        new tpm::TransposeOp(3, {0, 1, 2, {-1, 3}}, 2, TransposeOp::D2W));
    candidate_ops.emplace_back(
        new tpm::TransposeOp(2, {0, 1, {-1, 2}, 3}, -2, TransposeOp::D2H));
    candidate_ops.emplace_back(
        new tpm::TransposeOp(3, {0, 1, 2, {-1, 3}}, -2, TransposeOp::D2W));
    candidate_ops.emplace_back(
        new tpm::ConvOp(conv->getPaddingMode(), 1, 1, 1, 1));
    candidate_ops.emplace_back(
        new tpm::ConvOp(conv->getPaddingMode(), 1, 1, 2, 1));
    candidate_ops.emplace_back(
        new tpm::ConvOp(conv->getPaddingMode(), 1, 1, 1, 2));
    max_depth = 3;
}

void CMutator::addCandidateOpsForTransKernelConv(
    std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg) {
    // addCandidateOpsForNormalConv(candidate_ops, sg);
    auto conv = dynamic_cast<ConvOp *>(sg->getOperators()[0]);
    candidate_ops.emplace_back(new TransposeOp(-1, {0, 1, 3, 2}));
    candidate_ops.emplace_back(
        new tpm::ConvOp(conv->getPaddingMode(), 1, 1, 1, 1));
    max_depth = 4;
}

void CMutator::addCandidateOpsForGroupConv(
    std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg) {
    std::vector<int> fvec;
    for (auto conv : sg->getOperators()) {
        fvec.emplace_back(conv->getInputs(1)->getDims()[0]);
    }
    // gcd mode
    auto fgcd = gcd(fvec);
    std::vector<int> splitvec;
    for (auto i : fvec)
        splitvec.emplace_back(i / fgcd);
    // Add the origin conv
    // TODO: cannot new a conv op because we do not have same padding now
    candidate_ops.emplace_back(sg->getOperators()[0]->clone());
    candidate_ops.emplace_back(new ConcatOp(0));
    candidate_ops.emplace_back(new ConcatOp(1));
    candidate_ops.emplace_back(new SplitOp(1, splitvec));

    // TODO: max mode

    group_size = (int)fvec.size();
    max_depth = 4;
}

void CMutator::addCandidateOpsForTransposeGroupConv(
    std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg) {
    candidate_ops.emplace_back(new ConvOp(ConvOp::Same));
    candidate_ops.emplace_back(new ConcatOp(0));
    candidate_ops.emplace_back(new ConcatOp(1));
    candidate_ops.emplace_back(
        new SplitOp(1, std::vector<int>(sg->getOperators().size(), 1)));
    candidate_ops.emplace_back(new TransposeOp(-1, {0, 1, 3, 2}));
    max_depth = 4;
}

void CMutator::addCandidateOpsForNormalMatmul(
    std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg) {
    candidate_ops.emplace_back(new MatmulOp(true, false));
    candidate_ops.emplace_back(new MatmulOp(true, true));
    candidate_ops.emplace_back(new MatmulOp(false, false));
    candidate_ops.emplace_back(new MatmulOp(false, true));
    candidate_ops.emplace_back(new tpm::TransposeOp(-1, {0, 2, 1}));
    max_depth = 4;
}
void CMutator::addCandidateOpsForBatchMatmul(
    std::vector<std::shared_ptr<Operator>> &candidate_ops, SubGraph *sg) {}

void CMutator::addPreprocessForGroupConvGCD(SubGraph *sg) {
    EQOPT
    // max_depth = 1;
    std::vector<int> fvec;
    for (auto conv : sg->getOperators()) {
        fvec.emplace_back(conv->getInputs(1)->getDims()[0]);
    }
    auto fgcd = gcd(fvec);
    reserveTensors(group_size * 4 + max_depth * 2);
    TensorVec ins, weis;
    for (size_t i = 0, iEnd = sg->getOperators().size(); i < iEnd; ++i) {
        for (size_t j = 0, jEnd = sg->getInputs().size(); j < jEnd; ++j) {
            auto input = searchingGraph->getTensors()[j];
            if (input->getHash() ==
                    sg->getOperators()[i]->getInputs(0)->getHash() &&
                input->getType() == Tensor::Input) {
                if (fvec[i] / fgcd > 1) {
                    auto extend = dynamic_cast<Operator *>(
                        new ExtendOp(1, fvec[i] / fgcd - 1));
                    Tensor *output = newTensor();

                    // TODO: why the function in base class cannot be called?
                    if (!extend->computeShape({input}, {output})) {
                        popBackTensor(extend);
                        continue;
                    }
                    pushBackOp(extend);

                    ins.emplace_back(output);
                } else {
                    ins.emplace_back(input);
                }
            }
            auto weight = searchingGraph->getTensors()[j];
            if (weight->getHash() ==
                sg->getOperators()[i]->getInputs(1)->getHash()) {
                weis.emplace_back(weight);
            }
        }
    }
    auto concat_in = dynamic_cast<Operator *>(new ConcatOp(1));
    Tensor *conv_in = newTensor();
    if (!concat_in->computeShape(ins, {conv_in})) {
        popBackTensor(concat_in);
        return;
    }
    pushBackOp(concat_in);
    auto concat_wei = dynamic_cast<Operator *>(new ConcatOp(0));
    Tensor *conv_wei = newTensor();
    if (!concat_wei->computeShape(weis, {conv_wei})) {
        popBackTensor(concat_wei);
        return;
    }
    pushBackOp(concat_wei);
    auto conv = dynamic_cast<Operator *>(new ConvOp(ConvOp::Same));
    Tensor *conv_out = newTensor();
    if (!conv->computeShape({conv_in, conv_wei}, {conv_out})) {
        popBackTensor(conv);
        return;
    }
    pushBackOp(conv);
    std::vector<int> splitvec;
    TensorVec splitouts;
    for (auto i : fvec) {
        splitvec.emplace_back(i / fgcd);
        splitouts.emplace_back(newTensor());
    }
    auto split = dynamic_cast<Operator *>(new SplitOp(1, splitvec));
    if (!split->computeShape({conv_out}, splitouts)) {
        for (size_t i = 0, iEnd = fvec.size(); i < iEnd; ++i) {
            popBackTensor(split);
        }
        return;
    }
    pushBackOp(split);
}

void CMutator::addPreprocessForGroupConvMAX(SubGraph *sg) {
    NEQOPT
    // max_depth = 1;
    std::vector<int> fvec;
    for (auto conv : sg->getOperators()) {
        fvec.emplace_back(conv->getInputs(1)->getDims()[0]);
    }
    auto fmax = max(fvec);
    reserveTensors(group_size * 4 + max_depth * 2);
    TensorVec ins, weis;
    int padOpSize = sg->getOperators()[0]->getInputs()[0]->getDims().size();
    for (size_t i = 0, iEnd = sg->getOperators().size(); i < iEnd; ++i) {
        for (size_t j = 0, jEnd = sg->getInputs().size(); j < jEnd; ++j) {
            auto weight = searchingGraph->getTensors()[j];
            if (weight->getHash() ==
                sg->getOperators()[i]->getInputs(1)->getHash()) {
                if (fvec[i] < fmax) {
                    //  && input->getType() == Tensor::Input) {
                    std::vector<int> padOpBegin(padOpSize, 0);
                    std::vector<int> padOpEnd(padOpSize, 0);
                    padOpEnd[0] = fmax - fvec[i];
                    auto zeros = dynamic_cast<Operator *>(
                        new PadOp(padOpBegin, padOpEnd));
                    Tensor *output = newTensor();

                    // TODO: why the function in base class cannot be
                    // called?
                    if (!zeros->computeShape({weight}, {output})) {
                        popBackTensor(zeros);
                        continue;
                    }
                    pushBackOp(zeros);

                    weis.emplace_back(output);
                } else {
                    weis.emplace_back(weight);
                }
            }
            auto input = searchingGraph->getTensors()[j];
            if (input->getHash() ==
                sg->getOperators()[i]->getInputs(0)->getHash()) {
                ins.emplace_back(input);
            }
        }
    }
    auto concat_in = dynamic_cast<Operator *>(new ConcatOp(1));
    Tensor *conv_in = newTensor();
    if (!concat_in->computeShape(ins, {conv_in})) {
        popBackTensor(concat_in);
        return;
    }
    pushBackOp(concat_in);
    auto concat_wei = dynamic_cast<Operator *>(new ConcatOp(0));
    Tensor *conv_wei = newTensor();
    if (!concat_wei->computeShape(weis, {conv_wei})) {
        popBackTensor(concat_wei);
        return;
    }
    pushBackOp(concat_wei);
    auto conv = dynamic_cast<Operator *>(new ConvOp(ConvOp::Same));
    Tensor *conv_out = newTensor();
    if (!conv->computeShape({conv_in, conv_wei}, {conv_out})) {
        popBackTensor(conv);
        return;
    }
    pushBackOp(conv);
    std::vector<int> splitvec;
    TensorVec splitouts;
    for (auto i : fvec) {
        if (i < fmax) {
            splitvec.emplace_back(i);
            splitvec.emplace_back(fmax - i);
            splitouts.emplace_back(newTensor());
            auto zeroTensor = newTensor();
            zeroTensor->setType(Tensor::NotCounted);
            splitouts.emplace_back(zeroTensor);
        } else {
            splitvec.emplace_back(i);
            splitouts.emplace_back(newTensor());
        }
    }
    auto split = dynamic_cast<Operator *>(new SplitOp(1, splitvec));
    if (!split->computeShape({conv_out}, splitouts)) {
        for (size_t i = 0, iEnd = fvec.size(); i < iEnd; ++i) {
            popBackTensor(split);
        }
        return;
    }
    pushBackOp(split);
}

void CMutator::addPreprocessForGroupConvOneInput(SubGraph *sg) {
    EQOPT
    // max_depth = 1;
    std::vector<int> fvec;
    for (auto conv : sg->getOperators()) {
        fvec.emplace_back(conv->getInputs(1)->getDims()[0]);
    }
    reserveTensors(group_size * 4 + max_depth * 2);
    TensorVec ins, weis;
    for (size_t i = 0, iEnd = sg->getOperators().size(); i < iEnd; ++i) {
        for (size_t j = 0, jEnd = sg->getInputs().size(); j < jEnd; ++j) {
            auto weight = searchingGraph->getTensors()[j];
            if (weight->getHash() ==
                sg->getOperators()[i]->getInputs(1)->getHash()) {
                weis.emplace_back(weight);
            }
            auto input = searchingGraph->getTensors()[j];
            if (input->getHash() ==
                sg->getOperators()[0]->getInputs(0)->getHash()) {
                ins.emplace_back(input);
            }
        }
    }
    if (ins.size() != weis.size())
        return;
    auto in0 = ins[0];
    auto concat_wei = dynamic_cast<Operator *>(new ConcatOp(0));
    Tensor *conv_wei = newTensor();
    if (!concat_wei->computeShape(weis, {conv_wei})) {
        popBackTensor(concat_wei);
        return;
    }
    pushBackOp(concat_wei);
    auto conv = dynamic_cast<Operator *>(new ConvOp(ConvOp::Same));
    Tensor *conv_out = newTensor();
    if (!conv->computeShape({in0, conv_wei}, {conv_out})) {
        popBackTensor(conv);
        return;
    }
    pushBackOp(conv);
    std::vector<int> splitvec;
    TensorVec splitouts;
    for (auto i : fvec) {
        splitvec.emplace_back(i);
        splitouts.emplace_back(newTensor());
    }
    auto split = dynamic_cast<Operator *>(new SplitOp(1, splitvec));
    if (!split->computeShape({conv_out}, splitouts)) {
        for (size_t i = 0, iEnd = fvec.size(); i < iEnd; ++i) {
            popBackTensor(split);
        }
        return;
    }
    pushBackOp(split);
}

void CMutator::addPreprocessForPadSlice(SubGraph *sg) {
    NEQOPT
    // max_depth = 1;
    auto input = searchingGraph->getTensors()[0];
    auto weight = searchingGraph->getTensors()[1];
    auto pad = dynamic_cast<Operator *>(new PadOp({0, 0, 0, 0}, {0, 0, 1, 1}));
    Tensor *i1 = newTensor();
    if (!pad->computeShape({input}, {i1})) {
        popBackTensor(pad);
        return;
    }
    pushBackOp(pad);
    auto conv = dynamic_cast<Operator *>(sg->getOperators()[0]->clone());
    Tensor *i2 = newTensor();
    if (!conv->computeShape({i1, weight}, {i2})) {
        popBackTensor(conv);
        return;
    }
    pushBackOp(conv);
    auto slice =
        dynamic_cast<Operator *>(new SliceOp({0, 0, 0, 0}, {0, 0, 1, 1}));
    Tensor *i3 = newTensor();
    if (!slice->computeShape({i2}, {i3})) {
        popBackTensor(slice);
        return;
    }
    pushBackOp(slice);
}

void CMutator::addPreprocessForConv1x1(SubGraph *sg) {
    EQOPT
    // max_depth = 1;
    auto t0 = newTensor(), t1 = newTensor(), t2 = newTensor();
    auto conv = sg->getOperators()[0];
    auto input_tensor = searchingGraph->getTensors()[0];
    auto weight_tensor = searchingGraph->getTensors()[1];
    auto input_dims = input_tensor->getDims();
    int f = conv->getInputs()[1]->getDims()[0];

    t0->setDims({input_dims[0], f, input_dims[1]}); // gemm weight
    t0->setType(Tensor::Weight);
    t1->setDims({input_dims[0], input_dims[1], input_dims[2] * input_dims[3]});

    Operator *reshape_weight = nullptr, *concat_weight = nullptr;
    if (input_dims[0] == 1) {
        reshape_weight = (new ReshapeOp({weight_tensor}, {t0}));
        pushBackOp(reshape_weight);
    } else {
        auto tt0 = newTensor();
        tt0->setDims({1, f, input_dims[1]}); // gemm weight
        tt0->setType(Tensor::Weight);
        reshape_weight = (new ReshapeOp({weight_tensor}, {tt0}));
        TensorVec concat_inputs(input_dims[0], tt0);
        concat_weight = (new ConcatOp(concat_inputs, t0, 0));
        pushBackOp(reshape_weight);
        pushBackOp(concat_weight);
    }
    auto reshape_input =
        dynamic_cast<Operator *>(new ReshapeOp({input_tensor}, {t1}));
    pushBackOp(reshape_input);

    auto matmul = dynamic_cast<Operator *>(new MatmulOp(false, false));
    if (!matmul->computeShape({t0, t1}, {t2})) {
        popBackTensor(matmul);
        if (input_dims[0] > 1)
            popBackTensor(concat_weight);
        popBackTensor(reshape_input);
        popBackTensor(reshape_weight);
        return;
    }
    pushBackOp(matmul);

    auto t3 = newTensor();
    auto reshape_output = dynamic_cast<Operator *>(new ReshapeOp({t2}, {t3}));
    t3->setDims(conv->getOutputs()[0]->getDims());
    pushBackOp(reshape_output);
}

void CMutator::addPreprocessForTransKernel(SubGraph *sg) {
    EQOPT
    // max_depth = 1;
    auto input = searchingGraph->getTensors()[0];
    auto weight = searchingGraph->getTensors()[1];
    auto itrans = dynamic_cast<Operator *>(new TransposeOp(-1, {0, 1, 3, 2}));
    auto t1 = newTensor();
    if (!itrans->computeShape({input}, {t1})) {
        popBackTensor(itrans);
        return;
    }
    pushBackOp(itrans);
    auto wtrans = dynamic_cast<Operator *>(new TransposeOp(-1, {0, 1, 3, 2}));
    auto t2 = newTensor();
    if (!wtrans->computeShape({weight}, {t2})) {
        popBackTensor(wtrans);
        return;
    }
    pushBackOp(wtrans);
    auto conv = dynamic_cast<Operator *>(sg->getOperators()[0]->clone());
    auto t3 = newTensor();
    if (!conv->computeShape({t1, t2}, {t3})) {
        popBackTensor(conv);
        return;
    }
    pushBackOp(conv);
    auto otrans = dynamic_cast<Operator *>(new TransposeOp(-1, {0, 1, 3, 2}));
    auto t4 = newTensor();
    if (!otrans->computeShape({t3}, {t4})) {
        popBackTensor(otrans);
        return;
    }
    pushBackOp(otrans);
}
void CMutator::addPreprocessForBatchMatmul(SubGraph *sg) {
    EQOPT
    // max_depth = 1;
    // auto lhs = searchingGraph->getTensors()[0];
    // auto rhs = searchingGraph->getTensors()[1];
    std::vector<int> sizes;
    TensorVec lhs, rhs;
    auto first_gemm = dynamic_cast<MatmulOp *>(sg->getOperators()[0]);
    for (auto op : sg->getOperators()) {
        Tensor *left, *right;
        for (size_t i = 0, iEnd = sg->getInputs().size(); i < iEnd; ++i) {
            auto tensor = searchingGraph->getTensors()[i];
            if (tensor->getHash() == op->getInputs(0)->getHash())
                left = tensor;
            else if (tensor->getHash() == op->getInputs(1)->getHash())
                right = tensor;
        }
        lhs.emplace_back(left);
        rhs.emplace_back(right);
        sizes.emplace_back(left->getDims()[0]);
    }
    auto t1 = newTensor();
    auto concat1 = dynamic_cast<Operator *>(new ConcatOp(0));
    if (!concat1->computeShape(lhs, {t1})) {
        popBackTensor(concat1);
        return;
    }
    pushBackOp(concat1);

    auto t2 = newTensor();
    auto concat2 = dynamic_cast<Operator *>(new ConcatOp(0));
    if (!concat2->computeShape(rhs, {t2})) {
        popBackTensor(concat2);
        return;
    }
    pushBackOp(concat2);

    auto t3 = newTensor();
    auto matmul = dynamic_cast<Operator *>(
        new MatmulOp(first_gemm->getTransA(), first_gemm->getTransB()));

    if (!matmul->computeShape({t1, t2}, {t3})) {
        popBackTensor(matmul);
        return;
    }
    pushBackOp(matmul);

    TensorVec outs;
    for (size_t i = 0, iEnd = sizes.size(); i < iEnd; ++i)
        outs.emplace_back(newTensor());
    auto split = dynamic_cast<Operator *>(new SplitOp(0, sizes));
    if (!split->computeShape({t3}, outs)) {
        for (size_t i = 0, iEnd = sizes.size(); i < iEnd; ++i)
            popBackTensor(split);
        return;
    }
    pushBackOp(split);
}

uint64_t CMutator::computeHashForSingleComputeOp(const Operator *op) {
    // assert(op->getType() == Operator::Conv);
    if (op->getType() == Operator::Conv) {
        auto conv = dynamic_cast<const ConvOp *>(op);
        auto hash = conv->getHash();
        auto inputDim = conv->getInputs()[0]->getDims();
        auto weightDim = conv->getOutputs()[0]->getDims();
        hash += inputDim[0] * 10000019 + inputDim[1] * 10000079 +
                inputDim[2] * 10000103 + inputDim[3] * 10000121 +
                weightDim[0] * 10000139 + weightDim[1] * 10000141 +
                weightDim[2] * 10000169 + weightDim[3] * 10000189;
        return hash;
    } else if (op->getType() == Operator::ConvTrans) {
        auto convt = dynamic_cast<const ConvTransOp *>(op);
        auto hash = convt->getHash();
        auto inputDim = convt->getInputs()[0]->getDims();
        auto weightDim = convt->getOutputs()[0]->getDims();
        hash += inputDim[0] * 10000019 + inputDim[1] * 10000079 +
                inputDim[2] * 10000103 + inputDim[3] * 10000121 +
                weightDim[0] * 10000139 + weightDim[1] * 10000141 +
                weightDim[2] * 10000169 + weightDim[3] * 10000189;
        return hash;
    } else if (op->getType() == Operator::Matmul) {
        static uint64_t matmulhash = 0;
        return matmulhash++;
    } else {
        std::cout << "Unsupported operator when generating hash: "
                  << op->getName() << std::endl;
        assert(false);
        return 0;
    }
}

void CMutator::addPreprocessForTransposeGroupConvRS(SubGraph *sg) {
    EQOPT
    // max_depth = 1;
    auto w0dim = sg->getOperators()[0]->getInputs()[1]->getDims();
    auto r = w0dim[2];
    TensorVec inputs, weights;
    std::vector<bool> trans;
    for (size_t i = 0, iEnd = sg->getOperators().size(); i < iEnd; ++i) {
        Tensor *input, *weight;
        for (size_t j = 0, jEnd = sg->getInputs().size(); j < jEnd; ++j) {
            auto t = searchingGraph->getTensors()[j];
            if (t->getHash() ==
                sg->getOperators()[i]->getInputs(0)->getHash()) {
                input = t;
            }
            if (t->getHash() ==
                sg->getOperators()[i]->getInputs(1)->getHash()) {
                weight = t;
            }
        }
        if (weight->getDims()[2] == r) {
            inputs.emplace_back(input);
            weights.emplace_back(weight);
            trans.emplace_back(false);
        } else {
            auto itrans =
                dynamic_cast<Operator *>(new TransposeOp(-1, {0, 1, 3, 2}));
            auto t1 = newTensor();
            if (!itrans->computeShape({input}, {t1})) {
                popBackTensor(itrans);
                return;
            }
            pushBackOp(itrans);
            auto wtrans =
                dynamic_cast<Operator *>(new TransposeOp(-1, {0, 1, 3, 2}));
            auto t2 = newTensor();
            if (!wtrans->computeShape({weight}, {t2})) {
                popBackTensor(wtrans);
                return;
            }
            pushBackOp(wtrans);
            inputs.emplace_back(t1);
            weights.emplace_back(t2);
            trans.emplace_back(true);
        }
    }
    auto concat_in = dynamic_cast<Operator *>(new ConcatOp(1));
    Tensor *conv_in = newTensor();
    if (!concat_in->computeShape(inputs, {conv_in})) {
        popBackTensor(concat_in);
        return;
    }
    pushBackOp(concat_in);
    auto concat_wei = dynamic_cast<Operator *>(new ConcatOp(0));
    Tensor *conv_wei = newTensor();
    if (!concat_wei->computeShape(weights, {conv_wei})) {
        popBackTensor(concat_wei);
        return;
    }
    pushBackOp(concat_wei);
    auto conv = dynamic_cast<Operator *>(new ConvOp(
        dynamic_cast<ConvOp *>(sg->getOperators()[0])->getPaddingMode()));
    Tensor *conv_out = newTensor();
    if (!conv->computeShape({conv_in, conv_wei}, {conv_out})) {
        popBackTensor(conv);
        return;
    }
    pushBackOp(conv);
    auto split = dynamic_cast<Operator *>(
        new SplitOp(1, std::vector<int>(sg->getOperators().size(), 1)));
    TensorVec outputs;
    for (size_t i = 0, iEnd = sg->getOperators().size(); i < iEnd; ++i)
        outputs.emplace_back(newTensor());
    if (!split->computeShape({conv_out}, outputs)) {
        for (size_t i = 0, iEnd = sg->getOperators().size(); i < iEnd; ++i)
            popBackTensor(split);
        return;
    }
    pushBackOp(split);
    for (size_t i = 0, iEnd = outputs.size(); i < iEnd; ++i) {
        if (trans[i]) {
            auto otrans =
                dynamic_cast<Operator *>(new TransposeOp(-1, {0, 1, 3, 2}));
            auto t = newTensor();
            if (!otrans->computeShape({outputs[i]}, {t})) {
                popBackTensor(otrans);
                return;
            }
            pushBackOp(otrans);
        }
    }
}

void CMutator::addPreprocessForTransposeGroupConvSR(SubGraph *sg) {
    EQOPT
    // max_depth = 1;
    auto w0dim = sg->getOperators()[0]->getInputs()[1]->getDims();
    auto r = w0dim[2];
    TensorVec inputs, weights;
    std::vector<bool> trans;
    for (size_t i = 0, iEnd = sg->getOperators().size(); i < iEnd; ++i) {
        Tensor *input, *weight;
        for (size_t j = 0, jEnd = sg->getInputs().size(); j < jEnd; ++j) {
            auto t = searchingGraph->getTensors()[j];
            if (t->getHash() ==
                sg->getOperators()[i]->getInputs(0)->getHash()) {
                input = t;
            }
            if (t->getHash() ==
                sg->getOperators()[i]->getInputs(1)->getHash()) {
                weight = t;
            }
        }
        if (weight->getDims()[2] != r) {
            inputs.emplace_back(input);
            weights.emplace_back(weight);
            trans.emplace_back(false);
        } else {
            auto itrans =
                dynamic_cast<Operator *>(new TransposeOp(-1, {0, 1, 3, 2}));
            auto t1 = newTensor();
            if (!itrans->computeShape({input}, {t1})) {
                popBackTensor(itrans);
                return;
            }
            pushBackOp(itrans);
            auto wtrans =
                dynamic_cast<Operator *>(new TransposeOp(-1, {0, 1, 3, 2}));
            auto t2 = newTensor();
            if (!wtrans->computeShape({weight}, {t2})) {
                popBackTensor(wtrans);
                return;
            }
            pushBackOp(wtrans);
            inputs.emplace_back(t1);
            weights.emplace_back(t2);
            trans.emplace_back(true);
        }
    }
    auto concat_in = dynamic_cast<Operator *>(new ConcatOp(1));
    Tensor *conv_in = newTensor();
    if (!concat_in->computeShape(inputs, {conv_in})) {
        popBackTensor(concat_in);
        return;
    }
    pushBackOp(concat_in);
    auto concat_wei = dynamic_cast<Operator *>(new ConcatOp(0));
    Tensor *conv_wei = newTensor();
    if (!concat_wei->computeShape(weights, {conv_wei})) {
        popBackTensor(concat_wei);
        return;
    }
    pushBackOp(concat_wei);
    auto conv = dynamic_cast<Operator *>(new ConvOp(
        dynamic_cast<ConvOp *>(sg->getOperators()[0])->getPaddingMode()));
    Tensor *conv_out = newTensor();
    if (!conv->computeShape({conv_in, conv_wei}, {conv_out})) {
        popBackTensor(conv);
        return;
    }
    pushBackOp(conv);
    auto split = dynamic_cast<Operator *>(
        new SplitOp(1, std::vector<int>(sg->getOperators().size(), 1)));
    TensorVec outputs;
    for (size_t i = 0, iEnd = sg->getOperators().size(); i < iEnd; ++i)
        outputs.emplace_back(newTensor());
    if (!split->computeShape({conv_out}, outputs)) {
        for (size_t i = 0, iEnd = sg->getOperators().size(); i < iEnd; ++i)
            popBackTensor(split);
        return;
    }
    pushBackOp(split);
    for (size_t i = 0, iEnd = outputs.size(); i < iEnd; ++i) {
        if (trans[i]) {
            auto otrans =
                dynamic_cast<Operator *>(new TransposeOp(-1, {0, 1, 3, 2}));
            auto t = newTensor();
            if (!otrans->computeShape({outputs[i]}, {t})) {
                popBackTensor(otrans);
                return;
            }
            pushBackOp(otrans);
        }
    }
}

void CMutator::splitGroupConv(SubGraph *sg,
                              std::vector<SubGraph *> &out_graphs) {
    if (sg->getOperators().size() != 1)
        return;
    if (sg->getOperators()[0]->getType() != Operator::Conv)
        return;
    // auto op = sg->getOperators()[0];
    // assume ordered
    auto input = searchingGraph->getTensors()[0],
         weight = searchingGraph->getTensors()[1];
    auto group = input->getDims()[1] / weight->getDims()[1] * 2;
    while (group % 4 == 0) {
        // TODO: move to clear graph
        while (!oplist.empty())
            popBackOp();
        while (num_valid_tensors > sg->getInputs().size())
            popBackTensor();
        group /= 2;
        std::vector<int> splitvec(group, 1);
        reserveTensors(num_valid_tensors + 5 * group);
        auto spi = dynamic_cast<Operator *>(new SplitOp(1, splitvec));
        TensorVec iouts;
        for (int i = 0; i < group; ++i)
            iouts.emplace_back(newTensor());
        if (!spi->computeShape({input}, iouts)) {
            for (int i = 0; i < group; ++i)
                popBackTensor(spi);
            return;
        }
        pushBackOp(spi);

        auto spw = dynamic_cast<Operator *>(new SplitOp(0, splitvec));
        TensorVec wouts;
        for (int i = 0; i < group; ++i)
            wouts.emplace_back(newTensor());
        if (!spw->computeShape({weight}, wouts)) {
            for (int i = 0; i < group; ++i)
                popBackTensor(spw);
        }
        pushBackOp(spw);

        TensorVec couts;
        for (int i = 0; i < group; ++i) {
            auto cin = iouts[i], cw = wouts[i];
            auto conv = dynamic_cast<Operator *>(
                new ConvOp(dynamic_cast<ConvOp *>(sg->getOperators()[0])
                               ->getPaddingMode()));
            auto cout = newTensor();
            if (!conv->computeShape({cin, cw}, {cout})) {
                popBackTensor(conv);
                return;
            }
            pushBackOp(conv);
            couts.emplace_back(cout);
        }
        auto concat = dynamic_cast<Operator *>(new ConcatOp(1));
        auto out = newTensor();
        if (!concat->computeShape(couts, {out})) {
            popBackTensor(concat);
            return;
        }
        pushBackOp(concat);
        searchingGraph->resetOps(oplist, num_valid_tensors);
        // if (is_a_mutant(searchingGraph, sg, false)) {
        //     SubGraph *new_graph = new SubGraph(oplist);
        //     out_graphs.emplace_back(new_graph);
        // }
        SubGraph *new_graph = new SubGraph(oplist);
        for (size_t i = 0, iEnd = new_graph->getOutputs().size(); i < iEnd; ++i)
            new_graph->getOutputs()[i]->clone(sg->getOutputs()[i]);
        if (validDepth(new_graph))
            out_graphs.emplace_back(new_graph);
    }
}

void CMutator::updateCache(SubGraph *sg, std::vector<SubGraph *> &out_graphs) {
    std::cout << "========== Update Cache ==========" << std::endl;
    // sg->print();
    // add to L1 cache
    auto l1CacheIdx = sg->getCacheIdx(1);
    if (l1_mutation_cache.find(l1CacheIdx) == l1_mutation_cache.end()) {
        auto l1Cache = std::vector<std::shared_ptr<SubGraph>>{};
        std::cout << "Update L1 Cache Index: " << l1CacheIdx << std::endl;
        for (auto out : out_graphs) {
            l1Cache.emplace_back(new SubGraph(out->getOperators()));
        }
        l1_mutation_cache.emplace(l1CacheIdx, l1Cache);
    } else {
        std::cout << "Already in L1 Cache" << std::endl;
    }

    // add to L2 cache
    auto l2CacheIdx = sg->getCacheIdx(2);
    if (l2_mutation_cache.find(l2CacheIdx) == l2_mutation_cache.end()) {
        auto l2Cache = std::vector<std::shared_ptr<SubGraph>>{};
        std::cout << "Update L2 Cache Index: " << l2CacheIdx << std::endl;
        for (auto out : out_graphs) {
            l2Cache.emplace_back(new SubGraph(out->getOperators()));
        }
        l2_mutation_cache.emplace(l2CacheIdx, l2Cache);
    } else {
        std::cout << "Already in L2 Cache" << std::endl;
    }

    // storeCache(l1CacheIdx, l1Cache);
    // storeCache(l2CacheIdx, l2Cache);
}

bool CMutator::checkCache(SubGraph *sg, std::vector<SubGraph *> &out_graphs) {
    std::cout << "========== Check Cache ==========" << std::endl;
    // sg->print();
    auto l1CacheIdx = sg->getCacheIdx(1);
    auto l2CacheIdx = sg->getCacheIdx(2);
    // first check l1 cache, then check l2 cache
    if (l1_mutation_cache.find(l1CacheIdx) != l1_mutation_cache.end()) {
        std::cout << "Try L1 Cache" << std::endl;
        out_graphs.clear();
        for (auto mut : l1_mutation_cache[l1CacheIdx]) {
            auto new_graph = new SubGraph(mut->getOperators());
            markTransType(sg, new_graph);
            out_graphs.emplace_back(new_graph);
        }
        std::cout << "Hit L1 Cache!" << std::endl;
        return true;
    } else if (l2_mutation_cache.find(l2CacheIdx) != l2_mutation_cache.end()) {
        std::cout << "Try L2 Cache" << std::endl;
        out_graphs.clear();
        for (auto mut : l2_mutation_cache[l2CacheIdx]) {
            bool isError = false;
            if (sg->getInputs().size() != mut->getInputs().size()) {
                std::cout << "Not Hit L2 Cache: Do not Match" << std::endl;
                continue;
            }
            // construct out_graphs
            // is exhaustion meaningful ?
            auto &opList = mut->getOperators();
            OpVec newOpList;
            std::unordered_map<int, int> nodeMap;
            std::vector<int> cnt(opList.size());
            std::vector<int> q;
            for (size_t i = 0; i < opList.size(); i++) {
                auto &op = opList[i];
                nodeMap.emplace(op->getGuid(), i);
                cnt[i] = op->getPredecessors().size();
                if (op->getPredecessors().size() == 0) {
                    q.emplace_back(i);
                    // update subgraph inputs
                    TensorVec inputs;
                    TensorVec outputs;
                    for (size_t j = 0; j < op->getOutputs().size(); j++) {
                        outputs.emplace_back(newTensor());
                    }
                    for (size_t j = 0; j < op->getInputs().size(); j++) {
                        for (size_t k = 0; k < mut->getInputs().size(); k++) {
                            if (op->getInputs()[j]->getGuid() ==
                                mut->getInputs()[k]->getGuid()) {
                                sg->getInputs()[k]->setType(
                                    op->getInputs()[j]->getType());
                                inputs.emplace_back(sg->getInputs()[k]);
                                if (inputs.size() == op->getInputs().size()) {
                                    break;
                                }
                            }
                        }
                    }

                    if (!op->computeShape(inputs, outputs)) {
                        std::cout << "Not Hit L2 Cache: Shape Error"
                                  << std::endl;
                        // std::cout << "[ERROR] Op:" << std::endl;
                        // op->print();
                        // std::cout << "[ERROR] Op Inputs:" << inputs.size() <<
                        // std::endl; std::cout << "[ERROR] Op Outputs:" <<
                        // outputs.size() << std::endl;
                        isError = true;
                        break;
                    }
                    newOpList.emplace_back(op);
                }
            }

            if (isError) {
                continue;
            }

            for (int st = 0, ed = q.size(); st < ed; st++) {
                int id = q[st];
                auto &op = opList[id];

                for (auto suc : op->getSuccessors()) {
                    int suc_id = nodeMap[suc->getGuid()];
                    cnt[suc_id]--;
                    if (cnt[suc_id] == 0) {
                        q.emplace_back(suc_id);
                        ed++;
                    }
                }

                if (op->getPredecessors().size() == 0) {
                    continue;
                }

                TensorVec inputs;
                TensorVec outputs;
                for (size_t i = 0; i < op->getOutputs().size(); i++) {
                    outputs.emplace_back(newTensor());
                }
                for (size_t i = 0; i < op->getInputs().size(); i++) {
                    if (op->getInputs()[i]->getOutputOf() == nullptr) {
                        for (size_t j = 0; j < mut->getInputs().size(); j++) {
                            if (op->getInputs()[i]->getGuid() ==
                                mut->getInputs()[j]->getGuid()) {
                                sg->getInputs()[j]->setType(
                                    op->getInputs()[i]->getType());
                                inputs.emplace_back(sg->getInputs()[j]);
                                if (inputs.size() == op->getInputs().size()) {
                                    break;
                                }
                            }
                        }
                    } else {
                        for (size_t j = 0; j < opList.size(); j++) {
                            if (op->getInputs()[i]->getOutputOf()->getGuid() ==
                                opList[j]->getGuid()) {
                                for (size_t k = 0; k < newOpList.size(); k++) {
                                    if (newOpList[k]->getGuid() ==
                                        opList[j]->getGuid()) {
                                        // DEBUG: left or right
                                        newOpList[k]->getOutputs()[0]->setType(
                                            op->getInputs()[i]->getType());
                                        inputs.emplace_back(
                                            newOpList[k]->getOutputs()[0]);
                                        if (inputs.size() ==
                                            op->getInputs().size()) {
                                            break;
                                        }
                                    }
                                }
                            }
                            if (inputs.size() == op->getInputs().size()) {
                                break;
                            }
                        }
                    }
                }

                if (!op->computeShape(inputs, outputs)) {
                    std::cout << "Not Hit L2 Cache: Shape Error" << std::endl;
                    // std::cout << "[ERROR] Op:" << std::endl;
                    // op->print();
                    // std::cout << "[ERROR] Op Inputs:" << inputs.size() <<
                    // std::endl; std::cout << "[ERROR] Op Outputs:" <<
                    // outputs.size() << std::endl; std::cout << "[ERROR]
                    // OpList:" << std::endl; for (auto op : opList) {
                    //     op->print();
                    // }
                    // std::cout << "[ERROR] NewOpList:" << std::endl;
                    // for (auto op : newOpList) {
                    //     op->print();
                    // }
                    isError = true;
                    break;
                }
                newOpList.emplace_back(op);
            }

            if (isError) {
                continue;
            }

            // verify out_graphs
            bool fullComputing = true;
            for (auto op : newOpList) {
                if (!op->isTransposeOp()) {
                    fullComputing = false;
                    break;
                }
            }

            auto newGraph = new SubGraph(newOpList);
            for (auto input : newGraph->getInputs()) {
                input->dataRand();
            }
            computingPos.clear();
            auto outputs = newGraph->getOutputs();
            for (size_t i = 0, iEnd = outputs.size(); i < iEnd; ++i) {
                computingPos.emplace_back(std::vector<std::pair<Dim, VType>>());
                auto &back = computingPos.back();
                auto dm = outputs[i]->getDims();
                srand(time(NULL));
                for (int j = 0; j < 8; ++j) {
                    Dim randPos = {};
                    for (auto d : dm)
                        randPos.emplace_back(((rand() % 2) + 1) * d / 3);
                    back.emplace_back(std::make_pair(randPos, 0));
                }
                for (auto &pos : back) {
                    auto comp = newGraph->compute(pos.first, i);
                    if (!comp.first) {
                        std::cout << "Not Hit L2 Cache: Verification Error"
                                  << std::endl;
                        isError = false;
                        break;
                    }
                    pos.second = comp.second;
                }
                if (isError) {
                    break;
                }
            }

            if (isError) {
                continue;
            }

            if (is_a_mutant(newGraph, sg, fullComputing)) {
                markTransType(sg, newGraph);
                out_graphs.emplace_back(newGraph);
                std::cout << "Hit L2 Cache!" << std::endl;
            } else {
                std::cout << "Not Hit L2 Cache: Not a Mutant" << std::endl;
            }
        }
        return out_graphs.size() ? true : false;
    } else {
        std::cout << "Not Hit Cache!" << std::endl;
        return false;
    }
}

bool CMutator::storeCache(uint64_t cacheIdx,
                          std::vector<std::shared_ptr<SubGraph>> out_graphs) {
    for (auto out : out_graphs) {
        std::string name = "cache_" + std::to_string(cacheIdx) + "_" +
                           std::to_string(out->getHash());
        out->exportOnnx(name.c_str());
    }
    return true;
}

// -std=c++17
// bool CMutator::loadCache(uint64_t cacheIdx, std::vector<SubGraph *>
// &out_graphs) {
//     out_graphs.clear();
//     std::string path = "./";
//     std::string name = "cache_" + std::to_string(cacheIdx);
//     for (const auto & entry : std::filesystem::directory_iterator(path)) {
//         if (entry.path().find(name) != string::npos) {
//             auto graph = new tpm::SubGraph();
//             graph->importOnnx(files[i].c_str());
//             out_graphs.emplace_back(graph);
//         }
//     }
//     return true;
// }

bool CMutator::loadCache(uint64_t cacheIdx,
                         std::vector<std::shared_ptr<SubGraph>> out_graphs) {
    out_graphs.clear();
    DIR *dir;
    struct dirent *ent;
    std::string name = "cache_123_456";
    if ((dir = opendir("./")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            if (ent->d_name == name) {
                auto graph = new tpm::SubGraph();
                graph->importOnnx(("./" + name).c_str());
                out_graphs.emplace_back(graph);
            }
        }
        closedir(dir);
        return true;
    }
    return false;
}

void CMutator::resetGraph(const SubGraph *in_graph) {
    while (!oplist.empty())
        popBackOp();
    while (num_valid_tensors > in_graph->getInputs().size())
        popBackTensor();
}

void CMutator::markTransType(SubGraph *inputGraph, SubGraph *outputGraph) {
    auto &ops = outputGraph->getOperators();
    if (ops.size() != 3) {
        return;
    }
    if (ops[1]->getType() == Operator::Conv) {
        if (ops[0]->getType() != Operator::Transpose) {
            return;
        }
        if (ops[2]->getType() != Operator::Transpose) {
            return;
        }
        auto preTrans = dynamic_cast<TransposeOp *>(ops[0]);
        auto conv = dynamic_cast<ConvOp *>(ops[1]);
        auto postTrans = dynamic_cast<TransposeOp *>(ops[2]);
        preTrans->setPos(TransposeOp::Pre);
        postTrans->setPos(TransposeOp::Post);

        auto ph = conv->getPh(), pw = conv->getPw();
        auto prepenalty = preTrans->getInputs()[0]->getPenalty();
        switch (preTrans->getType()) {
        case TransposeOp::H2N: {
            // zero paddings do not need to copy
            // if (penalty_h > 0)
            //     penalty_h = (penalty_h - ph) / 2;
            // else
            // prepenalty[2] += ph;
            break;
        }
        case TransposeOp::N2H:
        case TransposeOp::D2H: {
            prepenalty[2] = prepenalty[2] * 2 + ph;
            break;
        }
        case TransposeOp::W2N: {
            // zero paddings do not need to copy
            // if (penalty_w > 0)
            //     penalty_w = (penalty_w - pw) / 2;
            // else
            // prepenalty[3] += pw;
            break;
        }
        case TransposeOp::N2W:
        case TransposeOp::D2W: {
            prepenalty[3] = prepenalty[3] * 2 + pw;
            break;
        }
        default:
            break;
        }
        preTrans->getInputs()[0]->clone(inputGraph->getInputs()[0]);
        preTrans->getOutputs()[0]->setPenalty(prepenalty);
        auto postpenalty = conv->computeOutputPenalty(prepenalty);
        postTrans->getInputs()[0]->setPenalty(postpenalty);
        postTrans->getOutputs()[0]->clone(inputGraph->getOutputs()[0]);
        return;
    }
    if (ops[2]->getType() == Operator::Conv) {
        if (ops[0]->getType() != Operator::Transpose) {
            return;
        }
        if (ops[1]->getType() != Operator::Transpose) {
        }
        auto conv = dynamic_cast<ConvOp *>(ops[2]);
        conv->getOutputs()[0]->setPenalty(
            conv->getInputs()[0]->getOutputOf()->getInputs()[0]->getPenalty());
        conv->getInputs()[0]->setPenalty(
            conv->getInputs()[0]->getOutputOf()->getInputs()[0]->getPenalty());
    }
}

bool CMutator::validDepth(SubGraph *sg) {
    if (sg->getOperators().size() > (size_t)max_depth && max_depth < 5)
        return false;
    return true;
}
