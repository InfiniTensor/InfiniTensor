#pragma once

#include "graph.h"

Graph split_each(UniGraph &&, std::function<float(UniGraph const &)> const &);
