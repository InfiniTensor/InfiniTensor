#include "nnet/Visitor/FullPrinterVisitor.h"
#include "nnet/Visitor/HashVisitor.h"
#include "nnet/Visitor/Serializer.h"
#include "nnet/expr.h"
using namespace nnet;
using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <log>\n", argv[0]);
        return 1;
    }
    auto expr = Serializer().fromFile(argv[1]);
    cout << FullPrinterVisitor().print(expr);
    cout << endl << "Hash = " << HashVisitor().getHash(expr) << endl;
    return 0;
}
