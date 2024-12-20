CXX = nvcc
CXXFLAGS = -std=c++17 -rdc=true
CXXFLAGS_OPT = -O3
DEBUGFLAGS = -g -DDEBUG=1
SRC_DIR = src
OBJ_DIR = obj
TARGET = regression.out
DEBUG_TARGET = regression_debug.out

REGRESSION_SRCS = regression.cpp $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
REGRESSION_OBJS = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(REGRESSION_SRCS)))
REGRESSION_OBJS := $(patsubst %.cu, $(OBJ_DIR)/%.o, $(REGRESSION_OBJS))

all: $(TARGET)

regression.out: $(REGRESSION_OBJS)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_OPT) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_OPT) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_OPT) -c $< -o $@

$(OBJ_DIR)/regression.o: regression.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_OPT) -c $< -o $@

debug: CXXFLAGS += $(DEBUGFLAGS)
debug: CXXFLAGS_OPT = -O0
debug: $(DEBUG_TARGET)

# removes -DDEBUG=1 from the flags
debug-noprint: DEBUGFLAGS = -g
debug-noprint: CXXFLAGS += $(DEBUGFLAGS)
debug-noprint: CXXFLAGS_OPT = -O0
debug-noprint: $(DEBUG_TARGET)

regression_debug.out: $(REGRESSION_OBJS)
	$(CXX) $(CXXFLAGS) $(CXXFLAGS_OPT) -o $@ $^

clean:
	rm -rf $(OBJ_DIR) *.out

.PHONY: all debug clean
