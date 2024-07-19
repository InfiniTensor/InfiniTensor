.PHONY : build clean all

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))
INSTALL_DIR := /usr/local/Ascend/ascend-toolkit/latest
RT_INC := $(INSTALL_DIR)/runtime/include
RT_LIB := $(INSTALL_DIR)/runtime/lib64

BISHENG_CMD := bisheng -O2 --cce-soc-version=Ascend910B3 --cce-soc-core-type=VecCore --shared -fPIC -I$(RT_INC) -L$(RT_LIB) -lascendcl -lruntime -I$(INSTALL_DIR)/compiler/tikcpp/tikcfw/ -I$(INSTALL_DIR)/compiler/tikcpp/tikcfw/impl -I$(INSTALL_DIR)/compiler/tikcpp/tikcfw/interface  --std=c++17 
# AR := ar 
# ARFLAGS := rcs
# $(wildcard $(MKFILE_DIR)*.cce)
CCE_FILES = plugin.cce
OBJ_FILES = libascend_plugin.so 

# libascend_plugin.so: $(OBJ_FILES)  
# 	$(AR) $(ARFLAGS) $@ $^ 

# %.o: %.cce  
# 	$(BISHENG_CMD) $< -o $@ 

$(OBJ_FILES): $(CCE_FILES)
	$(BISHENG_CMD) $^ -o $@

build: libascend_plugin.so
clean :
	rm -f libascend_plugin.a
	rm -f *.o
	rm -f *.so

SOURCE = demo_add.cce  
LIBRARY = libdemo_add.a
# 默认目标  
all: $(LIBRARY)  
  
# 编译规则  
$(LIBRARY): $(SOURCE)  
	$(BISHENG_CMD) $(SOURCE) -c -o $@ 