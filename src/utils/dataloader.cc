#include "utils/dataloader.h"
#include<fstream>

namespace infini {

    void loadTensorData(std::string file, int line) {
        std::ifstream filein;
        filein.open(file,std::ios::in);
        filein.close();

    }

    void saveTensorData(std::string file, int line) {
        std::ofstream fileout;
        fileout.open(file, std::ios::out);
        fileout.close();

    }

}; // namespace infini

