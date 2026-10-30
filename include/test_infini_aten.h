#pragma once
#include <cstdlib>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <sstream>
#include <stdexcept>

namespace infini {
#if INFINITENSOR_TEST_HAS_ATEN
inline PyThreadState *testAtenMainState = nullptr;
class TestAtenEnvironment : public ::testing::Environment {
    void TearDown() override {
        if (testAtenMainState) {
            // Run before main-thread TLS (including vendor stream tables) is
            // destroyed, and after every test has released its runtime.
            PyEval_RestoreThread(testAtenMainState);
            testAtenMainState = nullptr;
            pybind11::finalize_interpreter();
        }
    }
};
[[maybe_unused]] inline auto *testAtenEnvironment =
    ::testing::AddGlobalTestEnvironment(new TestAtenEnvironment());
#endif
// Python entry points load these modules before running ATen providers. Give
// standalone C++ integration tests the same environment, including vendor
// lazy initialization callbacks that require an active Python interpreter.
inline void initializeTestAtenProviders() {
#if INFINITENSOR_TEST_HAS_ATEN
    static const bool initialized = [] {
        const char *modules = std::getenv("INFINIOPS_PROVIDER_MODULES");
        if (!modules || !*modules)
            return true;
        const bool owned = !Py_IsInitialized();
        if (owned) {
            const char *executable =
                std::getenv("INFINITENSOR_TEST_PYTHON_EXECUTABLE");
#ifdef INFINITENSOR_TEST_PYTHON_DEFAULT
            if (!executable || !*executable)
                executable = INFINITENSOR_TEST_PYTHON_DEFAULT;
#endif
            if (!executable || !*executable)
                throw std::runtime_error(
                    "Set INFINITENSOR_TEST_PYTHON_EXECUTABLE to the build "
                    "Python");
            PyConfig config;
            PyConfig_InitPythonConfig(&config);
            const auto status = PyConfig_SetBytesString(
                &config, &config.program_name, executable);
            if (PyStatus_Exception(status)) {
                const std::string error = status.err_msg
                                              ? status.err_msg
                                              : "Invalid Python executable";
                PyConfig_Clear(&config);
                throw std::runtime_error(error);
            }
            pybind11::initialize_interpreter(&config);
        }
        {
            pybind11::gil_scoped_acquire gil;
            pybind11::module_::import("torch");
            std::istringstream names(modules);
            for (std::string name; std::getline(names, name, ',');)
                if (!name.empty())
                    pybind11::module_::import(name.c_str());
        }
        if (owned)
            testAtenMainState = PyEval_SaveThread();
        return true;
    }();
    (void)initialized;
#endif
}
} // namespace infini
