#define NOMINMAX
#include "xMatrix.h"
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <spdlog/spdlog.h>

constexpr int M = 1024;
constexpr int N = 1024;
constexpr int K = 1024;

TEST_CASE("Matrix Multiply", "[Multiply]") {
    auto A = xMatrixf::makeRandMat(M, N);
    auto B = xMatrixf::makeRandMat(N, K);
    xMatrixf C1(M, K);

    SECTION("execute multi1") {
        auto start = std::chrono::high_resolution_clock::now();
        multi1(*A, *B, C1);
        auto end = std::chrono::high_resolution_clock::now();
        spdlog::info("multi1 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    multi1(*A, *B, C1);
    SECTION("execute multi2") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi2(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();

        REQUIRE(isEqual(C1, C2));

        spdlog::info("multi2 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi3") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi3(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();

        REQUIRE(isEqual(C1, C2));

        spdlog::info("multi3 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi4") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi4(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));

        spdlog::info("multi4 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi5") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi5(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();

        REQUIRE(isEqual(C1, C2));

        spdlog::info("multi5 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi6") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi6(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi6 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi7") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi7(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi7 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi8") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi8(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi8 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi9") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi9(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi9 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi10") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi10(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi10 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi11") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi11(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi11 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi12") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi12(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi12 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi13") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi13(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi13 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi14") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi14(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi14 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi15") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi15(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi15 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi16") {
        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi16(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi16 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi17") {
        warnup();

        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi17(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));
        spdlog::info("multi17 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
    SECTION("execute multi18") {
        warnup();

        xMatrixf C2(M, K);
        auto start = std::chrono::high_resolution_clock::now();
        multi18(*A, *B, C2);
        auto end = std::chrono::high_resolution_clock::now();
        REQUIRE(isEqual(C1, C2));

        spdlog::info("multi18 execution time: {}ms",
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
}
