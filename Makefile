.PHONY: all clean run

# include directories
INCLUDE = inc/

# compiler options
INCLUDE_OPTS = $(foreach d, $(INCLUDE), -I$d)
CXX = g++
CXX_FLAGS = -std=c++11 -Wall -Werror $(INCLUDE_OPTS)

# output files and dependency lists
TRAIN_OUT_FILE = train.out
TRAIN_OUT_DEPS = src/train.cpp src/neural.cpp inc/neural.hpp
TEST_OUT_FILE = test.out
TEST_OUT_DEPS = src/test.cpp src/neural.cpp inc/neural.hpp


# make targets
all: $(TRAIN_OUT_FILE) $(TEST_OUT_FILE)

$(TRAIN_OUT_FILE): $(TRAIN_OUT_DEPS)
	@echo "Building '$(TRAIN_OUT_FILE)' ..."
	@$(CXX) $(CXX_FLAGS) -o $(TRAIN_OUT_FILE) $^

$(TEST_OUT_FILE): $(TEST_OUT_DEPS)
	@echo "Building '$(TEST_OUT_FILE)' ..."
	@$(CXX) $(CXX_FLAGS) -o $(TEST_OUT_FILE) $^

clean:
	@echo "Cleaning built files ..."
	rm -f *.o *.out

run: $(TRAIN_OUT_FILE) $(TEST_OUT_FILE)
	@./$(TRAIN_OUT_FILE)
	@./$(TEST_OUT_FILE)
