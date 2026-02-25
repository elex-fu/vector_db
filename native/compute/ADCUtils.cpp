#include "ADCUtils.h"
#include "DistanceUtils.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define HAVE_AVX2 1
#endif

namespace vectordb {

float adcDistanceScalar(const float* distanceTable, const uint8_t* codes, int pqM, int nCentroids) {
    float dist = 0.0f;
    for (int m = 0; m < pqM; m++) {
        dist += distanceTable[m * nCentroids + codes[m]];
    }
    return dist;
}

#if defined(HAVE_AVX2)

float adcDistanceAVX2(const float* distanceTable, const uint8_t* codes, int pqM, int nCentroids) {
    // Process 8 subspaces at a time using AVX2
    __m256 sumVec = _mm256_setzero_ps();

    int m = 0;
    // Process 8 subspaces at a time
    for (; m + 8 <= pqM; m += 8) {
        // Load 8 code indices
        uint32_t idx0 = codes[m];
        uint32_t idx1 = codes[m + 1];
        uint32_t idx2 = codes[m + 2];
        uint32_t idx3 = codes[m + 3];
        uint32_t idx4 = codes[m + 4];
        uint32_t idx5 = codes[m + 5];
        uint32_t idx6 = codes[m + 6];
        uint32_t idx7 = codes[m + 7];

        // Gather 8 float values from distance table
        // distanceTable layout: [pqM][nCentroids]
        const float* tableBase0 = distanceTable + (m + 0) * nCentroids;
        const float* tableBase1 = distanceTable + (m + 1) * nCentroids;
        const float* tableBase2 = distanceTable + (m + 2) * nCentroids;
        const float* tableBase3 = distanceTable + (m + 3) * nCentroids;
        const float* tableBase4 = distanceTable + (m + 4) * nCentroids;
        const float* tableBase5 = distanceTable + (m + 5) * nCentroids;
        const float* tableBase6 = distanceTable + (m + 6) * nCentroids;
        const float* tableBase7 = distanceTable + (m + 7) * nCentroids;

        // Load the 8 distance values
        __m256 distVec = _mm256_set_ps(
            tableBase7[idx7], tableBase6[idx6], tableBase5[idx5], tableBase4[idx4],
            tableBase3[idx3], tableBase2[idx2], tableBase1[idx1], tableBase0[idx0]
        );

        sumVec = _mm256_add_ps(sumVec, distVec);
    }

    // Horizontal sum of sumVec
    __m128 sumLow = _mm256_castps256_ps128(sumVec);
    __m128 sumHigh = _mm256_extractf128_ps(sumVec, 1);
    sumLow = _mm_add_ps(sumLow, sumHigh);
    sumLow = _mm_hadd_ps(sumLow, sumLow);
    sumLow = _mm_hadd_ps(sumLow, sumLow);
    float result = _mm_cvtss_f32(sumLow);

    // Process remaining subspaces
    for (; m < pqM; m++) {
        result += distanceTable[m * nCentroids + codes[m]];
    }

    return result;
}

void adcDistanceBatchAVX2(const float* distanceTable, const uint8_t* codes,
                          int nCodes, int pqM, int nCentroids, float* distances) {
    // Process multiple codes in parallel for better throughput
    const int batchSize = 8; // Process 8 codes at a time

    int c = 0;
    for (; c + batchSize <= nCodes; c += batchSize) {
        __m256 sumVecs[8];
        for (int i = 0; i < batchSize; i++) {
            sumVecs[i] = _mm256_setzero_ps();
        }

        // Process 8 subspaces at a time for all 8 codes
        int m = 0;
        for (; m + 8 <= pqM; m += 8) {
            // For each subspace, gather distances for all 8 codes
            for (int sub = 0; sub < 8; sub++) {
                const float* tableBase = distanceTable + (m + sub) * nCentroids;

                __m256 distVals = _mm256_set_ps(
                    tableBase[codes[(c + 7) * pqM + m + sub]],
                    tableBase[codes[(c + 6) * pqM + m + sub]],
                    tableBase[codes[(c + 5) * pqM + m + sub]],
                    tableBase[codes[(c + 4) * pqM + m + sub]],
                    tableBase[codes[(c + 3) * pqM + m + sub]],
                    tableBase[codes[(c + 2) * pqM + m + sub]],
                    tableBase[codes[(c + 1) * pqM + m + sub]],
                    tableBase[codes[(c + 0) * pqM + m + sub]]
                );

                // Accumulate - rotate which sumVec gets the values
                for (int i = 0; i < batchSize; i++) {
                    sumVecs[i] = _mm256_add_ps(sumVecs[i],
                        _mm256_permutevar8x32_ps(distVals, _mm256_set1_epi32(i)));
                }
            }
        }

        // Horizontal sum for each code
        for (int i = 0; i < batchSize; i++) {
            __m128 sumLow = _mm256_castps256_ps128(sumVecs[i]);
            __m128 sumHigh = _mm256_extractf128_ps(sumVecs[i], 1);
            sumLow = _mm_add_ps(sumLow, sumHigh);
            sumLow = _mm_hadd_ps(sumLow, sumLow);
            sumLow = _mm_hadd_ps(sumLow, sumLow);
            distances[c + i] = _mm_cvtss_f32(sumLow);
        }

        // Handle remaining subspaces for all 8 codes
        for (; m < pqM; m++) {
            for (int i = 0; i < batchSize; i++) {
                distances[c + i] += distanceTable[m * nCentroids + codes[(c + i) * pqM + m]];
            }
        }
    }

    // Handle remaining codes
    for (; c < nCodes; c++) {
        distances[c] = adcDistanceAVX2(distanceTable, codes + c * pqM, pqM, nCentroids);
    }
}

#else // No AVX2

float adcDistanceAVX2(const float* distanceTable, const uint8_t* codes, int pqM, int nCentroids) {
    return adcDistanceScalar(distanceTable, codes, pqM, nCentroids);
}

void adcDistanceBatchAVX2(const float* distanceTable, const uint8_t* codes,
                          int nCodes, int pqM, int nCentroids, float* distances) {
    for (int c = 0; c < nCodes; c++) {
        distances[c] = adcDistanceScalar(distanceTable, codes + c * pqM, pqM, nCentroids);
    }
}

#endif

ADCDistanceFunc getADCDistanceFunc() {
#if defined(HAVE_AVX2)
    if (detectISA() >= ISA::AVX2) {
        return adcDistanceAVX2;
    }
#endif
    return adcDistanceScalar;
}

ADCDistanceBatchFunc getADCDistanceBatchFunc() {
#if defined(HAVE_AVX2)
    if (detectISA() >= ISA::AVX2) {
        return adcDistanceBatchAVX2;
    }
#endif
    return [](const float* distanceTable, const uint8_t* codes, int nCodes, int pqM, int nCentroids, float* distances) {
        for (int c = 0; c < nCodes; c++) {
            distances[c] = adcDistanceScalar(distanceTable, codes + c * pqM, pqM, nCentroids);
        }
    };
}

} // namespace vectordb
