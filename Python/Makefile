CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

MOJI_SRCS = $(wildcard moji/cc/kernels/*.cc) $(wildcard moji/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

ZERO_OUT_TARGET_LIB = moji/python/ops/_zero_out_ops.so

# zero_out op for CPU
zero_out_op: $(ZERO_OUT_TARGET_LIB)

$(ZERO_OUT_TARGET_LIB): $(MOJI_SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

zero_out_test: moji/python/ops/zero_out_ops_test.py moji/python/ops/zero_out_ops.py $(ZERO_OUT_TARGET_LIB)
	$(PYTHON_BIN_PATH) moji/python/ops/zero_out_ops_test.py

zero_out_pip_pkg: $(ZERO_OUT_TARGET_LIB)
	./build_pip_pkg.sh make artifacts
