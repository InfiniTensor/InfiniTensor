#ifndef CUSTOM_OPS_CUH
#define CUSTOM_OPS_CUH

#include <cassert>

namespace tpm {

#ifdef _WIN32
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;
using int64_t = long long;
using uint64_t = unsigned long long;
#else
#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short
#define int64_t long long
#define uint64_t unsigned long long
#endif

inline __global__ void __launch_bounds__(58)
    sg2bmm_bs1_n10000_m64_w1000_d1_kernel0(float *__restrict__ q,
                                           float *__restrict__ k,
                                           float *__restrict__ SG2BMM) {
    float SG2BMM_local[120];
    __shared__ float q_shared[320];
    __shared__ float Kpad_shared[664];
    SG2BMM_local[(0)] = 0.000000e+00f;
    SG2BMM_local[(3)] = 0.000000e+00f;
    SG2BMM_local[(6)] = 0.000000e+00f;
    SG2BMM_local[(9)] = 0.000000e+00f;
    SG2BMM_local[(12)] = 0.000000e+00f;
    SG2BMM_local[(15)] = 0.000000e+00f;
    SG2BMM_local[(18)] = 0.000000e+00f;
    SG2BMM_local[(21)] = 0.000000e+00f;
    SG2BMM_local[(24)] = 0.000000e+00f;
    SG2BMM_local[(27)] = 0.000000e+00f;
    SG2BMM_local[(30)] = 0.000000e+00f;
    SG2BMM_local[(33)] = 0.000000e+00f;
    SG2BMM_local[(36)] = 0.000000e+00f;
    SG2BMM_local[(39)] = 0.000000e+00f;
    SG2BMM_local[(42)] = 0.000000e+00f;
    SG2BMM_local[(45)] = 0.000000e+00f;
    SG2BMM_local[(48)] = 0.000000e+00f;
    SG2BMM_local[(51)] = 0.000000e+00f;
    SG2BMM_local[(54)] = 0.000000e+00f;
    SG2BMM_local[(57)] = 0.000000e+00f;
    SG2BMM_local[(60)] = 0.000000e+00f;
    SG2BMM_local[(63)] = 0.000000e+00f;
    SG2BMM_local[(66)] = 0.000000e+00f;
    SG2BMM_local[(69)] = 0.000000e+00f;
    SG2BMM_local[(72)] = 0.000000e+00f;
    SG2BMM_local[(75)] = 0.000000e+00f;
    SG2BMM_local[(78)] = 0.000000e+00f;
    SG2BMM_local[(81)] = 0.000000e+00f;
    SG2BMM_local[(84)] = 0.000000e+00f;
    SG2BMM_local[(87)] = 0.000000e+00f;
    SG2BMM_local[(90)] = 0.000000e+00f;
    SG2BMM_local[(93)] = 0.000000e+00f;
    SG2BMM_local[(96)] = 0.000000e+00f;
    SG2BMM_local[(99)] = 0.000000e+00f;
    SG2BMM_local[(102)] = 0.000000e+00f;
    SG2BMM_local[(105)] = 0.000000e+00f;
    SG2BMM_local[(108)] = 0.000000e+00f;
    SG2BMM_local[(111)] = 0.000000e+00f;
    SG2BMM_local[(114)] = 0.000000e+00f;
    SG2BMM_local[(117)] = 0.000000e+00f;
    SG2BMM_local[(1)] = 0.000000e+00f;
    SG2BMM_local[(4)] = 0.000000e+00f;
    SG2BMM_local[(7)] = 0.000000e+00f;
    SG2BMM_local[(10)] = 0.000000e+00f;
    SG2BMM_local[(13)] = 0.000000e+00f;
    SG2BMM_local[(16)] = 0.000000e+00f;
    SG2BMM_local[(19)] = 0.000000e+00f;
    SG2BMM_local[(22)] = 0.000000e+00f;
    SG2BMM_local[(25)] = 0.000000e+00f;
    SG2BMM_local[(28)] = 0.000000e+00f;
    SG2BMM_local[(31)] = 0.000000e+00f;
    SG2BMM_local[(34)] = 0.000000e+00f;
    SG2BMM_local[(37)] = 0.000000e+00f;
    SG2BMM_local[(40)] = 0.000000e+00f;
    SG2BMM_local[(43)] = 0.000000e+00f;
    SG2BMM_local[(46)] = 0.000000e+00f;
    SG2BMM_local[(49)] = 0.000000e+00f;
    SG2BMM_local[(52)] = 0.000000e+00f;
    SG2BMM_local[(55)] = 0.000000e+00f;
    SG2BMM_local[(58)] = 0.000000e+00f;
    SG2BMM_local[(61)] = 0.000000e+00f;
    SG2BMM_local[(64)] = 0.000000e+00f;
    SG2BMM_local[(67)] = 0.000000e+00f;
    SG2BMM_local[(70)] = 0.000000e+00f;
    SG2BMM_local[(73)] = 0.000000e+00f;
    SG2BMM_local[(76)] = 0.000000e+00f;
    SG2BMM_local[(79)] = 0.000000e+00f;
    SG2BMM_local[(82)] = 0.000000e+00f;
    SG2BMM_local[(85)] = 0.000000e+00f;
    SG2BMM_local[(88)] = 0.000000e+00f;
    SG2BMM_local[(91)] = 0.000000e+00f;
    SG2BMM_local[(94)] = 0.000000e+00f;
    SG2BMM_local[(97)] = 0.000000e+00f;
    SG2BMM_local[(100)] = 0.000000e+00f;
    SG2BMM_local[(103)] = 0.000000e+00f;
    SG2BMM_local[(106)] = 0.000000e+00f;
    SG2BMM_local[(109)] = 0.000000e+00f;
    SG2BMM_local[(112)] = 0.000000e+00f;
    SG2BMM_local[(115)] = 0.000000e+00f;
    SG2BMM_local[(118)] = 0.000000e+00f;
    SG2BMM_local[(2)] = 0.000000e+00f;
    SG2BMM_local[(5)] = 0.000000e+00f;
    SG2BMM_local[(8)] = 0.000000e+00f;
    SG2BMM_local[(11)] = 0.000000e+00f;
    SG2BMM_local[(14)] = 0.000000e+00f;
    SG2BMM_local[(17)] = 0.000000e+00f;
    SG2BMM_local[(20)] = 0.000000e+00f;
    SG2BMM_local[(23)] = 0.000000e+00f;
    SG2BMM_local[(26)] = 0.000000e+00f;
    SG2BMM_local[(29)] = 0.000000e+00f;
    SG2BMM_local[(32)] = 0.000000e+00f;
    SG2BMM_local[(35)] = 0.000000e+00f;
    SG2BMM_local[(38)] = 0.000000e+00f;
    SG2BMM_local[(41)] = 0.000000e+00f;
    SG2BMM_local[(44)] = 0.000000e+00f;
    SG2BMM_local[(47)] = 0.000000e+00f;
    SG2BMM_local[(50)] = 0.000000e+00f;
    SG2BMM_local[(53)] = 0.000000e+00f;
    SG2BMM_local[(56)] = 0.000000e+00f;
    SG2BMM_local[(59)] = 0.000000e+00f;
    SG2BMM_local[(62)] = 0.000000e+00f;
    SG2BMM_local[(65)] = 0.000000e+00f;
    SG2BMM_local[(68)] = 0.000000e+00f;
    SG2BMM_local[(71)] = 0.000000e+00f;
    SG2BMM_local[(74)] = 0.000000e+00f;
    SG2BMM_local[(77)] = 0.000000e+00f;
    SG2BMM_local[(80)] = 0.000000e+00f;
    SG2BMM_local[(83)] = 0.000000e+00f;
    SG2BMM_local[(86)] = 0.000000e+00f;
    SG2BMM_local[(89)] = 0.000000e+00f;
    SG2BMM_local[(92)] = 0.000000e+00f;
    SG2BMM_local[(95)] = 0.000000e+00f;
    SG2BMM_local[(98)] = 0.000000e+00f;
    SG2BMM_local[(101)] = 0.000000e+00f;
    SG2BMM_local[(104)] = 0.000000e+00f;
    SG2BMM_local[(107)] = 0.000000e+00f;
    SG2BMM_local[(110)] = 0.000000e+00f;
    SG2BMM_local[(113)] = 0.000000e+00f;
    SG2BMM_local[(116)] = 0.000000e+00f;
    SG2BMM_local[(119)] = 0.000000e+00f;
    for (int p_outer_outer = 0; p_outer_outer < 16; ++p_outer_outer) {
        __syncthreads();
        q_shared[(((int)threadIdx.x))] =
            q[((((((((int)blockIdx.x) / 23) * 5120) +
                  ((((int)threadIdx.x) >> 2) * 64)) +
                 (p_outer_outer * 4)) +
                (((int)threadIdx.x) & 3)))];
        q_shared[((((int)threadIdx.x) + 58))] =
            q[((((((((int)blockIdx.x) / 23) * 5120) +
                  (((((int)threadIdx.x) + 58) >> 2) * 64)) +
                 (p_outer_outer * 4)) +
                ((((int)threadIdx.x) + 2) & 3)))];
        q_shared[((((int)threadIdx.x) + 116))] =
            q[(((((((((int)blockIdx.x) / 23) * 5120) +
                   ((((int)threadIdx.x) >> 2) * 64)) +
                  (p_outer_outer * 4)) +
                 (((int)threadIdx.x) & 3)) +
                1856))];
        q_shared[((((int)threadIdx.x) + 174))] =
            q[((((((((int)blockIdx.x) / 23) * 5120) +
                  (((((int)threadIdx.x) + 174) >> 2) * 64)) +
                 (p_outer_outer * 4)) +
                ((((int)threadIdx.x) + 2) & 3)))];
        q_shared[((((int)threadIdx.x) + 232))] =
            q[(((((((((int)blockIdx.x) / 23) * 5120) +
                   ((((int)threadIdx.x) >> 2) * 64)) +
                  (p_outer_outer * 4)) +
                 (((int)threadIdx.x) & 3)) +
                3712))];
        if (((int)threadIdx.x) < 30) {
            q_shared[((((int)threadIdx.x) + 290))] =
                q[((((((((int)blockIdx.x) / 23) * 5120) +
                      (((((int)threadIdx.x) + 290) >> 2) * 64)) +
                     (p_outer_outer * 4)) +
                    ((((int)threadIdx.x) + 2) & 3)))];
        }
        Kpad_shared[(((int)threadIdx.x))] =
            (((1000 <= ((((((int)blockIdx.x) % 23) * 87) +
                         (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                        (((int)threadIdx.x) >> 2))) &&
              (((((((int)blockIdx.x) % 23) * 87) +
                 (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                (((int)threadIdx.x) >> 2)) < 11000))
                 ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                           ((((int)blockIdx.x) / 23) * 5120)) +
                          ((((int)threadIdx.x) >> 2) * 64)) +
                         (p_outer_outer * 4)) +
                        (((int)threadIdx.x) & 3)) -
                       64000))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 58))] =
            (((1000 <= ((((((int)blockIdx.x) % 23) * 87) +
                         (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                        ((((int)threadIdx.x) + 58) >> 2))) &&
              (((((((int)blockIdx.x) % 23) * 87) +
                 (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                ((((int)threadIdx.x) + 58) >> 2)) < 11000))
                 ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                           ((((int)blockIdx.x) / 23) * 5120)) +
                          (((((int)threadIdx.x) + 58) >> 2) * 64)) +
                         (p_outer_outer * 4)) +
                        ((((int)threadIdx.x) + 2) & 3)) -
                       64000))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 116))] =
            (((971 <= ((((((int)blockIdx.x) % 23) * 87) +
                        (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                       (((int)threadIdx.x) >> 2))) &&
              (((((((int)blockIdx.x) % 23) * 87) +
                 (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                (((int)threadIdx.x) >> 2)) < 10971))
                 ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                           ((((int)blockIdx.x) / 23) * 5120)) +
                          ((((int)threadIdx.x) >> 2) * 64)) +
                         (p_outer_outer * 4)) +
                        (((int)threadIdx.x) & 3)) -
                       62144))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 174))] =
            (((1000 <= ((((((int)blockIdx.x) % 23) * 87) +
                         (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                        ((((int)threadIdx.x) + 174) >> 2))) &&
              (((((((int)blockIdx.x) % 23) * 87) +
                 (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                ((((int)threadIdx.x) + 174) >> 2)) < 11000))
                 ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                           ((((int)blockIdx.x) / 23) * 5120)) +
                          (((((int)threadIdx.x) + 174) >> 2) * 64)) +
                         (p_outer_outer * 4)) +
                        ((((int)threadIdx.x) + 2) & 3)) -
                       64000))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 232))] =
            (((942 <= ((((((int)blockIdx.x) % 23) * 87) +
                        (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                       (((int)threadIdx.x) >> 2))) &&
              (((((((int)blockIdx.x) % 23) * 87) +
                 (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                (((int)threadIdx.x) >> 2)) < 10942))
                 ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                           ((((int)blockIdx.x) / 23) * 5120)) +
                          ((((int)threadIdx.x) >> 2) * 64)) +
                         (p_outer_outer * 4)) +
                        (((int)threadIdx.x) & 3)) -
                       60288))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 290))] =
            (((1000 <= ((((((int)blockIdx.x) % 23) * 87) +
                         (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                        ((((int)threadIdx.x) + 290) >> 2))) &&
              (((((((int)blockIdx.x) % 23) * 87) +
                 (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                ((((int)threadIdx.x) + 290) >> 2)) < 11000))
                 ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                           ((((int)blockIdx.x) / 23) * 5120)) +
                          (((((int)threadIdx.x) + 290) >> 2) * 64)) +
                         (p_outer_outer * 4)) +
                        ((((int)threadIdx.x) + 2) & 3)) -
                       64000))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 348))] =
            (((913 <= ((((((int)blockIdx.x) % 23) * 87) +
                        (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                       (((int)threadIdx.x) >> 2))) &&
              (((((((int)blockIdx.x) % 23) * 87) +
                 (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                (((int)threadIdx.x) >> 2)) < 10913))
                 ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                           ((((int)blockIdx.x) / 23) * 5120)) +
                          ((((int)threadIdx.x) >> 2) * 64)) +
                         (p_outer_outer * 4)) +
                        (((int)threadIdx.x) & 3)) -
                       58432))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 406))] =
            (((1000 <= ((((((int)blockIdx.x) % 23) * 87) +
                         (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                        ((((int)threadIdx.x) + 406) >> 2))) &&
              (((((((int)blockIdx.x) % 23) * 87) +
                 (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                ((((int)threadIdx.x) + 406) >> 2)) < 11000))
                 ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                           ((((int)blockIdx.x) / 23) * 5120)) +
                          (((((int)threadIdx.x) + 406) >> 2) * 64)) +
                         (p_outer_outer * 4)) +
                        ((((int)threadIdx.x) + 2) & 3)) -
                       64000))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 464))] =
            (((884 <= ((((((int)blockIdx.x) % 23) * 87) +
                        (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                       (((int)threadIdx.x) >> 2))) &&
              (((((((int)blockIdx.x) % 23) * 87) +
                 (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                (((int)threadIdx.x) >> 2)) < 10884))
                 ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                           ((((int)blockIdx.x) / 23) * 5120)) +
                          ((((int)threadIdx.x) >> 2) * 64)) +
                         (p_outer_outer * 4)) +
                        (((int)threadIdx.x) & 3)) -
                       56576))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 522))] =
            (((1000 <= ((((((int)blockIdx.x) % 23) * 87) +
                         (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                        ((((int)threadIdx.x) + 522) >> 2))) &&
              (((((((int)blockIdx.x) % 23) * 87) +
                 (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                ((((int)threadIdx.x) + 522) >> 2)) < 11000))
                 ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                           ((((int)blockIdx.x) / 23) * 5120)) +
                          (((((int)threadIdx.x) + 522) >> 2) * 64)) +
                         (p_outer_outer * 4)) +
                        ((((int)threadIdx.x) + 2) & 3)) -
                       64000))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 580))] =
            (((855 <= ((((((int)blockIdx.x) % 23) * 87) +
                        (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                       (((int)threadIdx.x) >> 2))) &&
              (((((((int)blockIdx.x) % 23) * 87) +
                 (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                (((int)threadIdx.x) >> 2)) < 10855))
                 ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                           ((((int)blockIdx.x) / 23) * 5120)) +
                          ((((int)threadIdx.x) >> 2) * 64)) +
                         (p_outer_outer * 4)) +
                        (((int)threadIdx.x) & 3)) -
                       54720))]
                 : 0.000000e+00f);
        if (((int)threadIdx.x) < 26) {
            Kpad_shared[((((int)threadIdx.x) + 638))] =
                (((1000 <= ((((((int)blockIdx.x) % 23) * 87) +
                             (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                            ((((int)threadIdx.x) + 638) >> 2))) &&
                  (((((((int)blockIdx.x) % 23) * 87) +
                     (((((int)blockIdx.x) % 2875) / 23) * 80)) +
                    ((((int)threadIdx.x) + 638) >> 2)) < 11000))
                     ? k[((((((((((int)blockIdx.x) % 23) * 5568) +
                               ((((int)blockIdx.x) / 23) * 5120)) +
                              (((((int)threadIdx.x) + 638) >> 2) * 64)) +
                             (p_outer_outer * 4)) +
                            ((((int)threadIdx.x) + 2) & 3)) -
                           64000))]
                     : 0.000000e+00f);
        }
        __syncthreads();
        SG2BMM_local[(0)] =
            (SG2BMM_local[(0)] +
             (q_shared[(((((int)threadIdx.x) / 29) * 160))] *
              Kpad_shared[((((((int)threadIdx.x) / 29) * 160) +
                            ((((int)threadIdx.x) % 29) * 12)))]));
        SG2BMM_local[(3)] =
            (SG2BMM_local[(3)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 4))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            4))]));
        SG2BMM_local[(6)] =
            (SG2BMM_local[(6)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 8))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            8))]));
        SG2BMM_local[(9)] =
            (SG2BMM_local[(9)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 12))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            12))]));
        SG2BMM_local[(12)] =
            (SG2BMM_local[(12)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 16))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            16))]));
        SG2BMM_local[(15)] =
            (SG2BMM_local[(15)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 20))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            20))]));
        SG2BMM_local[(18)] =
            (SG2BMM_local[(18)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 24))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            24))]));
        SG2BMM_local[(21)] =
            (SG2BMM_local[(21)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 28))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            28))]));
        SG2BMM_local[(24)] =
            (SG2BMM_local[(24)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 32))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            32))]));
        SG2BMM_local[(27)] =
            (SG2BMM_local[(27)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 36))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            36))]));
        SG2BMM_local[(30)] =
            (SG2BMM_local[(30)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 40))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            40))]));
        SG2BMM_local[(33)] =
            (SG2BMM_local[(33)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 44))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            44))]));
        SG2BMM_local[(36)] =
            (SG2BMM_local[(36)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 48))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            48))]));
        SG2BMM_local[(39)] =
            (SG2BMM_local[(39)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 52))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            52))]));
        SG2BMM_local[(42)] =
            (SG2BMM_local[(42)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 56))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            56))]));
        SG2BMM_local[(45)] =
            (SG2BMM_local[(45)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 60))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            60))]));
        SG2BMM_local[(48)] =
            (SG2BMM_local[(48)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 64))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            64))]));
        SG2BMM_local[(51)] =
            (SG2BMM_local[(51)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 68))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            68))]));
        SG2BMM_local[(54)] =
            (SG2BMM_local[(54)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 72))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            72))]));
        SG2BMM_local[(57)] =
            (SG2BMM_local[(57)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 76))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            76))]));
        SG2BMM_local[(60)] =
            (SG2BMM_local[(60)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 80))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            80))]));
        SG2BMM_local[(63)] =
            (SG2BMM_local[(63)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 84))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            84))]));
        SG2BMM_local[(66)] =
            (SG2BMM_local[(66)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 88))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            88))]));
        SG2BMM_local[(69)] =
            (SG2BMM_local[(69)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 92))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            92))]));
        SG2BMM_local[(72)] =
            (SG2BMM_local[(72)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 96))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            96))]));
        SG2BMM_local[(75)] =
            (SG2BMM_local[(75)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 100))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            100))]));
        SG2BMM_local[(78)] =
            (SG2BMM_local[(78)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 104))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            104))]));
        SG2BMM_local[(81)] =
            (SG2BMM_local[(81)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 108))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            108))]));
        SG2BMM_local[(84)] =
            (SG2BMM_local[(84)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 112))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            112))]));
        SG2BMM_local[(87)] =
            (SG2BMM_local[(87)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 116))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            116))]));
        SG2BMM_local[(90)] =
            (SG2BMM_local[(90)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 120))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            120))]));
        SG2BMM_local[(93)] =
            (SG2BMM_local[(93)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 124))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            124))]));
        SG2BMM_local[(96)] =
            (SG2BMM_local[(96)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 128))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            128))]));
        SG2BMM_local[(99)] =
            (SG2BMM_local[(99)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 132))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            132))]));
        SG2BMM_local[(102)] =
            (SG2BMM_local[(102)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 136))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            136))]));
        SG2BMM_local[(105)] =
            (SG2BMM_local[(105)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 140))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            140))]));
        SG2BMM_local[(108)] =
            (SG2BMM_local[(108)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 144))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            144))]));
        SG2BMM_local[(111)] =
            (SG2BMM_local[(111)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 148))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            148))]));
        SG2BMM_local[(114)] =
            (SG2BMM_local[(114)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 152))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            152))]));
        SG2BMM_local[(117)] =
            (SG2BMM_local[(117)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 156))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            156))]));
        SG2BMM_local[(0)] =
            (SG2BMM_local[(0)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 1))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            1))]));
        SG2BMM_local[(3)] =
            (SG2BMM_local[(3)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 5))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            5))]));
        SG2BMM_local[(6)] =
            (SG2BMM_local[(6)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 9))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            9))]));
        SG2BMM_local[(9)] =
            (SG2BMM_local[(9)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 13))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            13))]));
        SG2BMM_local[(12)] =
            (SG2BMM_local[(12)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 17))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            17))]));
        SG2BMM_local[(15)] =
            (SG2BMM_local[(15)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 21))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            21))]));
        SG2BMM_local[(18)] =
            (SG2BMM_local[(18)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 25))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            25))]));
        SG2BMM_local[(21)] =
            (SG2BMM_local[(21)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 29))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            29))]));
        SG2BMM_local[(24)] =
            (SG2BMM_local[(24)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 33))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            33))]));
        SG2BMM_local[(27)] =
            (SG2BMM_local[(27)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 37))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            37))]));
        SG2BMM_local[(30)] =
            (SG2BMM_local[(30)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 41))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            41))]));
        SG2BMM_local[(33)] =
            (SG2BMM_local[(33)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 45))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            45))]));
        SG2BMM_local[(36)] =
            (SG2BMM_local[(36)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 49))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            49))]));
        SG2BMM_local[(39)] =
            (SG2BMM_local[(39)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 53))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            53))]));
        SG2BMM_local[(42)] =
            (SG2BMM_local[(42)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 57))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            57))]));
        SG2BMM_local[(45)] =
            (SG2BMM_local[(45)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 61))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            61))]));
        SG2BMM_local[(48)] =
            (SG2BMM_local[(48)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 65))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            65))]));
        SG2BMM_local[(51)] =
            (SG2BMM_local[(51)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 69))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            69))]));
        SG2BMM_local[(54)] =
            (SG2BMM_local[(54)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 73))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            73))]));
        SG2BMM_local[(57)] =
            (SG2BMM_local[(57)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 77))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            77))]));
        SG2BMM_local[(60)] =
            (SG2BMM_local[(60)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 81))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            81))]));
        SG2BMM_local[(63)] =
            (SG2BMM_local[(63)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 85))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            85))]));
        SG2BMM_local[(66)] =
            (SG2BMM_local[(66)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 89))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            89))]));
        SG2BMM_local[(69)] =
            (SG2BMM_local[(69)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 93))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            93))]));
        SG2BMM_local[(72)] =
            (SG2BMM_local[(72)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 97))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            97))]));
        SG2BMM_local[(75)] =
            (SG2BMM_local[(75)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 101))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            101))]));
        SG2BMM_local[(78)] =
            (SG2BMM_local[(78)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 105))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            105))]));
        SG2BMM_local[(81)] =
            (SG2BMM_local[(81)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 109))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            109))]));
        SG2BMM_local[(84)] =
            (SG2BMM_local[(84)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 113))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            113))]));
        SG2BMM_local[(87)] =
            (SG2BMM_local[(87)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 117))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            117))]));
        SG2BMM_local[(90)] =
            (SG2BMM_local[(90)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 121))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            121))]));
        SG2BMM_local[(93)] =
            (SG2BMM_local[(93)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 125))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            125))]));
        SG2BMM_local[(96)] =
            (SG2BMM_local[(96)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 129))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            129))]));
        SG2BMM_local[(99)] =
            (SG2BMM_local[(99)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 133))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            133))]));
        SG2BMM_local[(102)] =
            (SG2BMM_local[(102)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 137))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            137))]));
        SG2BMM_local[(105)] =
            (SG2BMM_local[(105)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 141))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            141))]));
        SG2BMM_local[(108)] =
            (SG2BMM_local[(108)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 145))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            145))]));
        SG2BMM_local[(111)] =
            (SG2BMM_local[(111)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 149))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            149))]));
        SG2BMM_local[(114)] =
            (SG2BMM_local[(114)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 153))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            153))]));
        SG2BMM_local[(117)] =
            (SG2BMM_local[(117)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 157))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            157))]));
        SG2BMM_local[(0)] =
            (SG2BMM_local[(0)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 2))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            2))]));
        SG2BMM_local[(3)] =
            (SG2BMM_local[(3)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 6))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            6))]));
        SG2BMM_local[(6)] =
            (SG2BMM_local[(6)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 10))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            10))]));
        SG2BMM_local[(9)] =
            (SG2BMM_local[(9)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 14))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            14))]));
        SG2BMM_local[(12)] =
            (SG2BMM_local[(12)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 18))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            18))]));
        SG2BMM_local[(15)] =
            (SG2BMM_local[(15)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 22))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            22))]));
        SG2BMM_local[(18)] =
            (SG2BMM_local[(18)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 26))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            26))]));
        SG2BMM_local[(21)] =
            (SG2BMM_local[(21)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 30))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            30))]));
        SG2BMM_local[(24)] =
            (SG2BMM_local[(24)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 34))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            34))]));
        SG2BMM_local[(27)] =
            (SG2BMM_local[(27)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 38))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            38))]));
        SG2BMM_local[(30)] =
            (SG2BMM_local[(30)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 42))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            42))]));
        SG2BMM_local[(33)] =
            (SG2BMM_local[(33)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 46))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            46))]));
        SG2BMM_local[(36)] =
            (SG2BMM_local[(36)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 50))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            50))]));
        SG2BMM_local[(39)] =
            (SG2BMM_local[(39)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 54))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            54))]));
        SG2BMM_local[(42)] =
            (SG2BMM_local[(42)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 58))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            58))]));
        SG2BMM_local[(45)] =
            (SG2BMM_local[(45)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 62))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            62))]));
        SG2BMM_local[(48)] =
            (SG2BMM_local[(48)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 66))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            66))]));
        SG2BMM_local[(51)] =
            (SG2BMM_local[(51)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 70))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            70))]));
        SG2BMM_local[(54)] =
            (SG2BMM_local[(54)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 74))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            74))]));
        SG2BMM_local[(57)] =
            (SG2BMM_local[(57)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 78))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            78))]));
        SG2BMM_local[(60)] =
            (SG2BMM_local[(60)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 82))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            82))]));
        SG2BMM_local[(63)] =
            (SG2BMM_local[(63)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 86))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            86))]));
        SG2BMM_local[(66)] =
            (SG2BMM_local[(66)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 90))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            90))]));
        SG2BMM_local[(69)] =
            (SG2BMM_local[(69)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 94))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            94))]));
        SG2BMM_local[(72)] =
            (SG2BMM_local[(72)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 98))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            98))]));
        SG2BMM_local[(75)] =
            (SG2BMM_local[(75)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 102))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            102))]));
        SG2BMM_local[(78)] =
            (SG2BMM_local[(78)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 106))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            106))]));
        SG2BMM_local[(81)] =
            (SG2BMM_local[(81)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 110))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            110))]));
        SG2BMM_local[(84)] =
            (SG2BMM_local[(84)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 114))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            114))]));
        SG2BMM_local[(87)] =
            (SG2BMM_local[(87)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 118))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            118))]));
        SG2BMM_local[(90)] =
            (SG2BMM_local[(90)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 122))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            122))]));
        SG2BMM_local[(93)] =
            (SG2BMM_local[(93)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 126))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            126))]));
        SG2BMM_local[(96)] =
            (SG2BMM_local[(96)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 130))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            130))]));
        SG2BMM_local[(99)] =
            (SG2BMM_local[(99)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 134))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            134))]));
        SG2BMM_local[(102)] =
            (SG2BMM_local[(102)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 138))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            138))]));
        SG2BMM_local[(105)] =
            (SG2BMM_local[(105)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 142))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            142))]));
        SG2BMM_local[(108)] =
            (SG2BMM_local[(108)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 146))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            146))]));
        SG2BMM_local[(111)] =
            (SG2BMM_local[(111)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 150))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            150))]));
        SG2BMM_local[(114)] =
            (SG2BMM_local[(114)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 154))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            154))]));
        SG2BMM_local[(117)] =
            (SG2BMM_local[(117)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 158))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            158))]));
        SG2BMM_local[(0)] =
            (SG2BMM_local[(0)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 3))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            3))]));
        SG2BMM_local[(3)] =
            (SG2BMM_local[(3)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 7))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            7))]));
        SG2BMM_local[(6)] =
            (SG2BMM_local[(6)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 11))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            11))]));
        SG2BMM_local[(9)] =
            (SG2BMM_local[(9)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 15))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            15))]));
        SG2BMM_local[(12)] =
            (SG2BMM_local[(12)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 19))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            19))]));
        SG2BMM_local[(15)] =
            (SG2BMM_local[(15)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 23))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            23))]));
        SG2BMM_local[(18)] =
            (SG2BMM_local[(18)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 27))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            27))]));
        SG2BMM_local[(21)] =
            (SG2BMM_local[(21)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 31))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            31))]));
        SG2BMM_local[(24)] =
            (SG2BMM_local[(24)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 35))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            35))]));
        SG2BMM_local[(27)] =
            (SG2BMM_local[(27)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 39))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            39))]));
        SG2BMM_local[(30)] =
            (SG2BMM_local[(30)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 43))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            43))]));
        SG2BMM_local[(33)] =
            (SG2BMM_local[(33)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 47))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            47))]));
        SG2BMM_local[(36)] =
            (SG2BMM_local[(36)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 51))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            51))]));
        SG2BMM_local[(39)] =
            (SG2BMM_local[(39)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 55))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            55))]));
        SG2BMM_local[(42)] =
            (SG2BMM_local[(42)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 59))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            59))]));
        SG2BMM_local[(45)] =
            (SG2BMM_local[(45)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 63))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            63))]));
        SG2BMM_local[(48)] =
            (SG2BMM_local[(48)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 67))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            67))]));
        SG2BMM_local[(51)] =
            (SG2BMM_local[(51)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 71))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            71))]));
        SG2BMM_local[(54)] =
            (SG2BMM_local[(54)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 75))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            75))]));
        SG2BMM_local[(57)] =
            (SG2BMM_local[(57)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 79))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            79))]));
        SG2BMM_local[(60)] =
            (SG2BMM_local[(60)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 83))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            83))]));
        SG2BMM_local[(63)] =
            (SG2BMM_local[(63)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 87))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            87))]));
        SG2BMM_local[(66)] =
            (SG2BMM_local[(66)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 91))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            91))]));
        SG2BMM_local[(69)] =
            (SG2BMM_local[(69)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 95))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            95))]));
        SG2BMM_local[(72)] =
            (SG2BMM_local[(72)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 99))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            99))]));
        SG2BMM_local[(75)] =
            (SG2BMM_local[(75)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 103))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            103))]));
        SG2BMM_local[(78)] =
            (SG2BMM_local[(78)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 107))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            107))]));
        SG2BMM_local[(81)] =
            (SG2BMM_local[(81)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 111))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            111))]));
        SG2BMM_local[(84)] =
            (SG2BMM_local[(84)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 115))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            115))]));
        SG2BMM_local[(87)] =
            (SG2BMM_local[(87)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 119))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            119))]));
        SG2BMM_local[(90)] =
            (SG2BMM_local[(90)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 123))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            123))]));
        SG2BMM_local[(93)] =
            (SG2BMM_local[(93)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 127))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            127))]));
        SG2BMM_local[(96)] =
            (SG2BMM_local[(96)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 131))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            131))]));
        SG2BMM_local[(99)] =
            (SG2BMM_local[(99)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 135))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            135))]));
        SG2BMM_local[(102)] =
            (SG2BMM_local[(102)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 139))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            139))]));
        SG2BMM_local[(105)] =
            (SG2BMM_local[(105)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 143))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            143))]));
        SG2BMM_local[(108)] =
            (SG2BMM_local[(108)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 147))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            147))]));
        SG2BMM_local[(111)] =
            (SG2BMM_local[(111)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 151))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            151))]));
        SG2BMM_local[(114)] =
            (SG2BMM_local[(114)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 155))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            155))]));
        SG2BMM_local[(117)] =
            (SG2BMM_local[(117)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 159))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            159))]));
        SG2BMM_local[(1)] = (SG2BMM_local[(1)] +
                             (q_shared[(((((int)threadIdx.x) / 29) * 160))] *
                              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                                             ((((int)threadIdx.x) % 29) * 12)) +
                                            4))]));
        SG2BMM_local[(4)] =
            (SG2BMM_local[(4)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 4))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            8))]));
        SG2BMM_local[(7)] =
            (SG2BMM_local[(7)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 8))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            12))]));
        SG2BMM_local[(10)] =
            (SG2BMM_local[(10)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 12))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            16))]));
        SG2BMM_local[(13)] =
            (SG2BMM_local[(13)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 16))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            20))]));
        SG2BMM_local[(16)] =
            (SG2BMM_local[(16)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 20))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            24))]));
        SG2BMM_local[(19)] =
            (SG2BMM_local[(19)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 24))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            28))]));
        SG2BMM_local[(22)] =
            (SG2BMM_local[(22)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 28))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            32))]));
        SG2BMM_local[(25)] =
            (SG2BMM_local[(25)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 32))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            36))]));
        SG2BMM_local[(28)] =
            (SG2BMM_local[(28)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 36))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            40))]));
        SG2BMM_local[(31)] =
            (SG2BMM_local[(31)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 40))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            44))]));
        SG2BMM_local[(34)] =
            (SG2BMM_local[(34)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 44))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            48))]));
        SG2BMM_local[(37)] =
            (SG2BMM_local[(37)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 48))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            52))]));
        SG2BMM_local[(40)] =
            (SG2BMM_local[(40)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 52))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            56))]));
        SG2BMM_local[(43)] =
            (SG2BMM_local[(43)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 56))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            60))]));
        SG2BMM_local[(46)] =
            (SG2BMM_local[(46)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 60))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            64))]));
        SG2BMM_local[(49)] =
            (SG2BMM_local[(49)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 64))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            68))]));
        SG2BMM_local[(52)] =
            (SG2BMM_local[(52)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 68))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            72))]));
        SG2BMM_local[(55)] =
            (SG2BMM_local[(55)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 72))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            76))]));
        SG2BMM_local[(58)] =
            (SG2BMM_local[(58)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 76))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            80))]));
        SG2BMM_local[(61)] =
            (SG2BMM_local[(61)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 80))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            84))]));
        SG2BMM_local[(64)] =
            (SG2BMM_local[(64)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 84))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            88))]));
        SG2BMM_local[(67)] =
            (SG2BMM_local[(67)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 88))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            92))]));
        SG2BMM_local[(70)] =
            (SG2BMM_local[(70)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 92))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            96))]));
        SG2BMM_local[(73)] =
            (SG2BMM_local[(73)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 96))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            100))]));
        SG2BMM_local[(76)] =
            (SG2BMM_local[(76)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 100))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            104))]));
        SG2BMM_local[(79)] =
            (SG2BMM_local[(79)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 104))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            108))]));
        SG2BMM_local[(82)] =
            (SG2BMM_local[(82)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 108))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            112))]));
        SG2BMM_local[(85)] =
            (SG2BMM_local[(85)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 112))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            116))]));
        SG2BMM_local[(88)] =
            (SG2BMM_local[(88)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 116))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            120))]));
        SG2BMM_local[(91)] =
            (SG2BMM_local[(91)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 120))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            124))]));
        SG2BMM_local[(94)] =
            (SG2BMM_local[(94)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 124))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            128))]));
        SG2BMM_local[(97)] =
            (SG2BMM_local[(97)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 128))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            132))]));
        SG2BMM_local[(100)] =
            (SG2BMM_local[(100)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 132))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            136))]));
        SG2BMM_local[(103)] =
            (SG2BMM_local[(103)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 136))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            140))]));
        SG2BMM_local[(106)] =
            (SG2BMM_local[(106)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 140))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            144))]));
        SG2BMM_local[(109)] =
            (SG2BMM_local[(109)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 144))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            148))]));
        SG2BMM_local[(112)] =
            (SG2BMM_local[(112)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 148))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            152))]));
        SG2BMM_local[(115)] =
            (SG2BMM_local[(115)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 152))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            156))]));
        SG2BMM_local[(118)] =
            (SG2BMM_local[(118)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 156))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            160))]));
        SG2BMM_local[(1)] =
            (SG2BMM_local[(1)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 1))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            5))]));
        SG2BMM_local[(4)] =
            (SG2BMM_local[(4)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 5))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            9))]));
        SG2BMM_local[(7)] =
            (SG2BMM_local[(7)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 9))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            13))]));
        SG2BMM_local[(10)] =
            (SG2BMM_local[(10)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 13))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            17))]));
        SG2BMM_local[(13)] =
            (SG2BMM_local[(13)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 17))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            21))]));
        SG2BMM_local[(16)] =
            (SG2BMM_local[(16)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 21))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            25))]));
        SG2BMM_local[(19)] =
            (SG2BMM_local[(19)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 25))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            29))]));
        SG2BMM_local[(22)] =
            (SG2BMM_local[(22)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 29))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            33))]));
        SG2BMM_local[(25)] =
            (SG2BMM_local[(25)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 33))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            37))]));
        SG2BMM_local[(28)] =
            (SG2BMM_local[(28)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 37))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            41))]));
        SG2BMM_local[(31)] =
            (SG2BMM_local[(31)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 41))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            45))]));
        SG2BMM_local[(34)] =
            (SG2BMM_local[(34)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 45))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            49))]));
        SG2BMM_local[(37)] =
            (SG2BMM_local[(37)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 49))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            53))]));
        SG2BMM_local[(40)] =
            (SG2BMM_local[(40)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 53))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            57))]));
        SG2BMM_local[(43)] =
            (SG2BMM_local[(43)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 57))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            61))]));
        SG2BMM_local[(46)] =
            (SG2BMM_local[(46)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 61))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            65))]));
        SG2BMM_local[(49)] =
            (SG2BMM_local[(49)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 65))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            69))]));
        SG2BMM_local[(52)] =
            (SG2BMM_local[(52)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 69))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            73))]));
        SG2BMM_local[(55)] =
            (SG2BMM_local[(55)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 73))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            77))]));
        SG2BMM_local[(58)] =
            (SG2BMM_local[(58)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 77))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            81))]));
        SG2BMM_local[(61)] =
            (SG2BMM_local[(61)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 81))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            85))]));
        SG2BMM_local[(64)] =
            (SG2BMM_local[(64)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 85))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            89))]));
        SG2BMM_local[(67)] =
            (SG2BMM_local[(67)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 89))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            93))]));
        SG2BMM_local[(70)] =
            (SG2BMM_local[(70)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 93))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            97))]));
        SG2BMM_local[(73)] =
            (SG2BMM_local[(73)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 97))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            101))]));
        SG2BMM_local[(76)] =
            (SG2BMM_local[(76)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 101))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            105))]));
        SG2BMM_local[(79)] =
            (SG2BMM_local[(79)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 105))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            109))]));
        SG2BMM_local[(82)] =
            (SG2BMM_local[(82)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 109))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            113))]));
        SG2BMM_local[(85)] =
            (SG2BMM_local[(85)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 113))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            117))]));
        SG2BMM_local[(88)] =
            (SG2BMM_local[(88)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 117))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            121))]));
        SG2BMM_local[(91)] =
            (SG2BMM_local[(91)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 121))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            125))]));
        SG2BMM_local[(94)] =
            (SG2BMM_local[(94)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 125))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            129))]));
        SG2BMM_local[(97)] =
            (SG2BMM_local[(97)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 129))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            133))]));
        SG2BMM_local[(100)] =
            (SG2BMM_local[(100)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 133))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            137))]));
        SG2BMM_local[(103)] =
            (SG2BMM_local[(103)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 137))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            141))]));
        SG2BMM_local[(106)] =
            (SG2BMM_local[(106)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 141))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            145))]));
        SG2BMM_local[(109)] =
            (SG2BMM_local[(109)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 145))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            149))]));
        SG2BMM_local[(112)] =
            (SG2BMM_local[(112)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 149))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            153))]));
        SG2BMM_local[(115)] =
            (SG2BMM_local[(115)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 153))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            157))]));
        SG2BMM_local[(118)] =
            (SG2BMM_local[(118)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 157))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            161))]));
        SG2BMM_local[(1)] =
            (SG2BMM_local[(1)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 2))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            6))]));
        SG2BMM_local[(4)] =
            (SG2BMM_local[(4)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 6))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            10))]));
        SG2BMM_local[(7)] =
            (SG2BMM_local[(7)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 10))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            14))]));
        SG2BMM_local[(10)] =
            (SG2BMM_local[(10)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 14))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            18))]));
        SG2BMM_local[(13)] =
            (SG2BMM_local[(13)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 18))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            22))]));
        SG2BMM_local[(16)] =
            (SG2BMM_local[(16)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 22))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            26))]));
        SG2BMM_local[(19)] =
            (SG2BMM_local[(19)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 26))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            30))]));
        SG2BMM_local[(22)] =
            (SG2BMM_local[(22)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 30))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            34))]));
        SG2BMM_local[(25)] =
            (SG2BMM_local[(25)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 34))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            38))]));
        SG2BMM_local[(28)] =
            (SG2BMM_local[(28)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 38))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            42))]));
        SG2BMM_local[(31)] =
            (SG2BMM_local[(31)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 42))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            46))]));
        SG2BMM_local[(34)] =
            (SG2BMM_local[(34)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 46))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            50))]));
        SG2BMM_local[(37)] =
            (SG2BMM_local[(37)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 50))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            54))]));
        SG2BMM_local[(40)] =
            (SG2BMM_local[(40)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 54))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            58))]));
        SG2BMM_local[(43)] =
            (SG2BMM_local[(43)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 58))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            62))]));
        SG2BMM_local[(46)] =
            (SG2BMM_local[(46)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 62))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            66))]));
        SG2BMM_local[(49)] =
            (SG2BMM_local[(49)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 66))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            70))]));
        SG2BMM_local[(52)] =
            (SG2BMM_local[(52)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 70))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            74))]));
        SG2BMM_local[(55)] =
            (SG2BMM_local[(55)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 74))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            78))]));
        SG2BMM_local[(58)] =
            (SG2BMM_local[(58)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 78))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            82))]));
        SG2BMM_local[(61)] =
            (SG2BMM_local[(61)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 82))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            86))]));
        SG2BMM_local[(64)] =
            (SG2BMM_local[(64)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 86))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            90))]));
        SG2BMM_local[(67)] =
            (SG2BMM_local[(67)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 90))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            94))]));
        SG2BMM_local[(70)] =
            (SG2BMM_local[(70)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 94))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            98))]));
        SG2BMM_local[(73)] =
            (SG2BMM_local[(73)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 98))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            102))]));
        SG2BMM_local[(76)] =
            (SG2BMM_local[(76)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 102))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            106))]));
        SG2BMM_local[(79)] =
            (SG2BMM_local[(79)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 106))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            110))]));
        SG2BMM_local[(82)] =
            (SG2BMM_local[(82)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 110))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            114))]));
        SG2BMM_local[(85)] =
            (SG2BMM_local[(85)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 114))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            118))]));
        SG2BMM_local[(88)] =
            (SG2BMM_local[(88)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 118))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            122))]));
        SG2BMM_local[(91)] =
            (SG2BMM_local[(91)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 122))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            126))]));
        SG2BMM_local[(94)] =
            (SG2BMM_local[(94)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 126))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            130))]));
        SG2BMM_local[(97)] =
            (SG2BMM_local[(97)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 130))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            134))]));
        SG2BMM_local[(100)] =
            (SG2BMM_local[(100)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 134))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            138))]));
        SG2BMM_local[(103)] =
            (SG2BMM_local[(103)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 138))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            142))]));
        SG2BMM_local[(106)] =
            (SG2BMM_local[(106)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 142))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            146))]));
        SG2BMM_local[(109)] =
            (SG2BMM_local[(109)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 146))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            150))]));
        SG2BMM_local[(112)] =
            (SG2BMM_local[(112)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 150))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            154))]));
        SG2BMM_local[(115)] =
            (SG2BMM_local[(115)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 154))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            158))]));
        SG2BMM_local[(118)] =
            (SG2BMM_local[(118)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 158))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            162))]));
        SG2BMM_local[(1)] =
            (SG2BMM_local[(1)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 3))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            7))]));
        SG2BMM_local[(4)] =
            (SG2BMM_local[(4)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 7))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            11))]));
        SG2BMM_local[(7)] =
            (SG2BMM_local[(7)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 11))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            15))]));
        SG2BMM_local[(10)] =
            (SG2BMM_local[(10)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 15))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            19))]));
        SG2BMM_local[(13)] =
            (SG2BMM_local[(13)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 19))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            23))]));
        SG2BMM_local[(16)] =
            (SG2BMM_local[(16)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 23))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            27))]));
        SG2BMM_local[(19)] =
            (SG2BMM_local[(19)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 27))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            31))]));
        SG2BMM_local[(22)] =
            (SG2BMM_local[(22)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 31))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            35))]));
        SG2BMM_local[(25)] =
            (SG2BMM_local[(25)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 35))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            39))]));
        SG2BMM_local[(28)] =
            (SG2BMM_local[(28)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 39))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            43))]));
        SG2BMM_local[(31)] =
            (SG2BMM_local[(31)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 43))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            47))]));
        SG2BMM_local[(34)] =
            (SG2BMM_local[(34)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 47))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            51))]));
        SG2BMM_local[(37)] =
            (SG2BMM_local[(37)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 51))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            55))]));
        SG2BMM_local[(40)] =
            (SG2BMM_local[(40)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 55))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            59))]));
        SG2BMM_local[(43)] =
            (SG2BMM_local[(43)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 59))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            63))]));
        SG2BMM_local[(46)] =
            (SG2BMM_local[(46)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 63))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            67))]));
        SG2BMM_local[(49)] =
            (SG2BMM_local[(49)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 67))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            71))]));
        SG2BMM_local[(52)] =
            (SG2BMM_local[(52)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 71))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            75))]));
        SG2BMM_local[(55)] =
            (SG2BMM_local[(55)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 75))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            79))]));
        SG2BMM_local[(58)] =
            (SG2BMM_local[(58)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 79))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            83))]));
        SG2BMM_local[(61)] =
            (SG2BMM_local[(61)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 83))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            87))]));
        SG2BMM_local[(64)] =
            (SG2BMM_local[(64)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 87))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            91))]));
        SG2BMM_local[(67)] =
            (SG2BMM_local[(67)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 91))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            95))]));
        SG2BMM_local[(70)] =
            (SG2BMM_local[(70)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 95))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            99))]));
        SG2BMM_local[(73)] =
            (SG2BMM_local[(73)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 99))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            103))]));
        SG2BMM_local[(76)] =
            (SG2BMM_local[(76)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 103))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            107))]));
        SG2BMM_local[(79)] =
            (SG2BMM_local[(79)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 107))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            111))]));
        SG2BMM_local[(82)] =
            (SG2BMM_local[(82)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 111))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            115))]));
        SG2BMM_local[(85)] =
            (SG2BMM_local[(85)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 115))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            119))]));
        SG2BMM_local[(88)] =
            (SG2BMM_local[(88)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 119))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            123))]));
        SG2BMM_local[(91)] =
            (SG2BMM_local[(91)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 123))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            127))]));
        SG2BMM_local[(94)] =
            (SG2BMM_local[(94)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 127))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            131))]));
        SG2BMM_local[(97)] =
            (SG2BMM_local[(97)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 131))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            135))]));
        SG2BMM_local[(100)] =
            (SG2BMM_local[(100)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 135))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            139))]));
        SG2BMM_local[(103)] =
            (SG2BMM_local[(103)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 139))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            143))]));
        SG2BMM_local[(106)] =
            (SG2BMM_local[(106)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 143))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            147))]));
        SG2BMM_local[(109)] =
            (SG2BMM_local[(109)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 147))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            151))]));
        SG2BMM_local[(112)] =
            (SG2BMM_local[(112)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 151))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            155))]));
        SG2BMM_local[(115)] =
            (SG2BMM_local[(115)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 155))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            159))]));
        SG2BMM_local[(118)] =
            (SG2BMM_local[(118)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 159))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            163))]));
        SG2BMM_local[(2)] = (SG2BMM_local[(2)] +
                             (q_shared[(((((int)threadIdx.x) / 29) * 160))] *
                              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                                             ((((int)threadIdx.x) % 29) * 12)) +
                                            8))]));
        SG2BMM_local[(5)] =
            (SG2BMM_local[(5)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 4))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            12))]));
        SG2BMM_local[(8)] =
            (SG2BMM_local[(8)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 8))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            16))]));
        SG2BMM_local[(11)] =
            (SG2BMM_local[(11)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 12))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            20))]));
        SG2BMM_local[(14)] =
            (SG2BMM_local[(14)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 16))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            24))]));
        SG2BMM_local[(17)] =
            (SG2BMM_local[(17)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 20))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            28))]));
        SG2BMM_local[(20)] =
            (SG2BMM_local[(20)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 24))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            32))]));
        SG2BMM_local[(23)] =
            (SG2BMM_local[(23)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 28))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            36))]));
        SG2BMM_local[(26)] =
            (SG2BMM_local[(26)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 32))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            40))]));
        SG2BMM_local[(29)] =
            (SG2BMM_local[(29)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 36))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            44))]));
        SG2BMM_local[(32)] =
            (SG2BMM_local[(32)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 40))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            48))]));
        SG2BMM_local[(35)] =
            (SG2BMM_local[(35)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 44))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            52))]));
        SG2BMM_local[(38)] =
            (SG2BMM_local[(38)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 48))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            56))]));
        SG2BMM_local[(41)] =
            (SG2BMM_local[(41)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 52))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            60))]));
        SG2BMM_local[(44)] =
            (SG2BMM_local[(44)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 56))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            64))]));
        SG2BMM_local[(47)] =
            (SG2BMM_local[(47)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 60))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            68))]));
        SG2BMM_local[(50)] =
            (SG2BMM_local[(50)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 64))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            72))]));
        SG2BMM_local[(53)] =
            (SG2BMM_local[(53)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 68))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            76))]));
        SG2BMM_local[(56)] =
            (SG2BMM_local[(56)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 72))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            80))]));
        SG2BMM_local[(59)] =
            (SG2BMM_local[(59)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 76))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            84))]));
        SG2BMM_local[(62)] =
            (SG2BMM_local[(62)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 80))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            88))]));
        SG2BMM_local[(65)] =
            (SG2BMM_local[(65)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 84))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            92))]));
        SG2BMM_local[(68)] =
            (SG2BMM_local[(68)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 88))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            96))]));
        SG2BMM_local[(71)] =
            (SG2BMM_local[(71)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 92))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            100))]));
        SG2BMM_local[(74)] =
            (SG2BMM_local[(74)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 96))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            104))]));
        SG2BMM_local[(77)] =
            (SG2BMM_local[(77)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 100))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            108))]));
        SG2BMM_local[(80)] =
            (SG2BMM_local[(80)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 104))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            112))]));
        SG2BMM_local[(83)] =
            (SG2BMM_local[(83)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 108))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            116))]));
        SG2BMM_local[(86)] =
            (SG2BMM_local[(86)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 112))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            120))]));
        SG2BMM_local[(89)] =
            (SG2BMM_local[(89)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 116))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            124))]));
        SG2BMM_local[(92)] =
            (SG2BMM_local[(92)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 120))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            128))]));
        SG2BMM_local[(95)] =
            (SG2BMM_local[(95)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 124))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            132))]));
        SG2BMM_local[(98)] =
            (SG2BMM_local[(98)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 128))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            136))]));
        SG2BMM_local[(101)] =
            (SG2BMM_local[(101)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 132))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            140))]));
        SG2BMM_local[(104)] =
            (SG2BMM_local[(104)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 136))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            144))]));
        SG2BMM_local[(107)] =
            (SG2BMM_local[(107)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 140))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            148))]));
        SG2BMM_local[(110)] =
            (SG2BMM_local[(110)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 144))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            152))]));
        SG2BMM_local[(113)] =
            (SG2BMM_local[(113)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 148))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            156))]));
        SG2BMM_local[(116)] =
            (SG2BMM_local[(116)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 152))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            160))]));
        SG2BMM_local[(119)] =
            (SG2BMM_local[(119)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 156))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            164))]));
        SG2BMM_local[(2)] =
            (SG2BMM_local[(2)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 1))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            9))]));
        SG2BMM_local[(5)] =
            (SG2BMM_local[(5)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 5))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            13))]));
        SG2BMM_local[(8)] =
            (SG2BMM_local[(8)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 9))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            17))]));
        SG2BMM_local[(11)] =
            (SG2BMM_local[(11)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 13))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            21))]));
        SG2BMM_local[(14)] =
            (SG2BMM_local[(14)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 17))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            25))]));
        SG2BMM_local[(17)] =
            (SG2BMM_local[(17)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 21))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            29))]));
        SG2BMM_local[(20)] =
            (SG2BMM_local[(20)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 25))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            33))]));
        SG2BMM_local[(23)] =
            (SG2BMM_local[(23)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 29))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            37))]));
        SG2BMM_local[(26)] =
            (SG2BMM_local[(26)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 33))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            41))]));
        SG2BMM_local[(29)] =
            (SG2BMM_local[(29)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 37))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            45))]));
        SG2BMM_local[(32)] =
            (SG2BMM_local[(32)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 41))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            49))]));
        SG2BMM_local[(35)] =
            (SG2BMM_local[(35)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 45))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            53))]));
        SG2BMM_local[(38)] =
            (SG2BMM_local[(38)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 49))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            57))]));
        SG2BMM_local[(41)] =
            (SG2BMM_local[(41)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 53))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            61))]));
        SG2BMM_local[(44)] =
            (SG2BMM_local[(44)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 57))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            65))]));
        SG2BMM_local[(47)] =
            (SG2BMM_local[(47)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 61))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            69))]));
        SG2BMM_local[(50)] =
            (SG2BMM_local[(50)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 65))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            73))]));
        SG2BMM_local[(53)] =
            (SG2BMM_local[(53)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 69))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            77))]));
        SG2BMM_local[(56)] =
            (SG2BMM_local[(56)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 73))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            81))]));
        SG2BMM_local[(59)] =
            (SG2BMM_local[(59)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 77))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            85))]));
        SG2BMM_local[(62)] =
            (SG2BMM_local[(62)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 81))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            89))]));
        SG2BMM_local[(65)] =
            (SG2BMM_local[(65)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 85))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            93))]));
        SG2BMM_local[(68)] =
            (SG2BMM_local[(68)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 89))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            97))]));
        SG2BMM_local[(71)] =
            (SG2BMM_local[(71)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 93))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            101))]));
        SG2BMM_local[(74)] =
            (SG2BMM_local[(74)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 97))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            105))]));
        SG2BMM_local[(77)] =
            (SG2BMM_local[(77)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 101))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            109))]));
        SG2BMM_local[(80)] =
            (SG2BMM_local[(80)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 105))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            113))]));
        SG2BMM_local[(83)] =
            (SG2BMM_local[(83)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 109))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            117))]));
        SG2BMM_local[(86)] =
            (SG2BMM_local[(86)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 113))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            121))]));
        SG2BMM_local[(89)] =
            (SG2BMM_local[(89)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 117))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            125))]));
        SG2BMM_local[(92)] =
            (SG2BMM_local[(92)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 121))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            129))]));
        SG2BMM_local[(95)] =
            (SG2BMM_local[(95)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 125))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            133))]));
        SG2BMM_local[(98)] =
            (SG2BMM_local[(98)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 129))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            137))]));
        SG2BMM_local[(101)] =
            (SG2BMM_local[(101)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 133))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            141))]));
        SG2BMM_local[(104)] =
            (SG2BMM_local[(104)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 137))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            145))]));
        SG2BMM_local[(107)] =
            (SG2BMM_local[(107)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 141))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            149))]));
        SG2BMM_local[(110)] =
            (SG2BMM_local[(110)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 145))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            153))]));
        SG2BMM_local[(113)] =
            (SG2BMM_local[(113)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 149))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            157))]));
        SG2BMM_local[(116)] =
            (SG2BMM_local[(116)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 153))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            161))]));
        SG2BMM_local[(119)] =
            (SG2BMM_local[(119)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 157))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            165))]));
        SG2BMM_local[(2)] =
            (SG2BMM_local[(2)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 2))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            10))]));
        SG2BMM_local[(5)] =
            (SG2BMM_local[(5)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 6))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            14))]));
        SG2BMM_local[(8)] =
            (SG2BMM_local[(8)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 10))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            18))]));
        SG2BMM_local[(11)] =
            (SG2BMM_local[(11)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 14))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            22))]));
        SG2BMM_local[(14)] =
            (SG2BMM_local[(14)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 18))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            26))]));
        SG2BMM_local[(17)] =
            (SG2BMM_local[(17)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 22))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            30))]));
        SG2BMM_local[(20)] =
            (SG2BMM_local[(20)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 26))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            34))]));
        SG2BMM_local[(23)] =
            (SG2BMM_local[(23)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 30))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            38))]));
        SG2BMM_local[(26)] =
            (SG2BMM_local[(26)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 34))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            42))]));
        SG2BMM_local[(29)] =
            (SG2BMM_local[(29)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 38))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            46))]));
        SG2BMM_local[(32)] =
            (SG2BMM_local[(32)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 42))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            50))]));
        SG2BMM_local[(35)] =
            (SG2BMM_local[(35)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 46))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            54))]));
        SG2BMM_local[(38)] =
            (SG2BMM_local[(38)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 50))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            58))]));
        SG2BMM_local[(41)] =
            (SG2BMM_local[(41)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 54))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            62))]));
        SG2BMM_local[(44)] =
            (SG2BMM_local[(44)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 58))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            66))]));
        SG2BMM_local[(47)] =
            (SG2BMM_local[(47)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 62))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            70))]));
        SG2BMM_local[(50)] =
            (SG2BMM_local[(50)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 66))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            74))]));
        SG2BMM_local[(53)] =
            (SG2BMM_local[(53)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 70))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            78))]));
        SG2BMM_local[(56)] =
            (SG2BMM_local[(56)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 74))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            82))]));
        SG2BMM_local[(59)] =
            (SG2BMM_local[(59)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 78))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            86))]));
        SG2BMM_local[(62)] =
            (SG2BMM_local[(62)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 82))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            90))]));
        SG2BMM_local[(65)] =
            (SG2BMM_local[(65)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 86))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            94))]));
        SG2BMM_local[(68)] =
            (SG2BMM_local[(68)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 90))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            98))]));
        SG2BMM_local[(71)] =
            (SG2BMM_local[(71)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 94))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            102))]));
        SG2BMM_local[(74)] =
            (SG2BMM_local[(74)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 98))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            106))]));
        SG2BMM_local[(77)] =
            (SG2BMM_local[(77)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 102))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            110))]));
        SG2BMM_local[(80)] =
            (SG2BMM_local[(80)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 106))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            114))]));
        SG2BMM_local[(83)] =
            (SG2BMM_local[(83)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 110))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            118))]));
        SG2BMM_local[(86)] =
            (SG2BMM_local[(86)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 114))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            122))]));
        SG2BMM_local[(89)] =
            (SG2BMM_local[(89)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 118))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            126))]));
        SG2BMM_local[(92)] =
            (SG2BMM_local[(92)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 122))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            130))]));
        SG2BMM_local[(95)] =
            (SG2BMM_local[(95)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 126))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            134))]));
        SG2BMM_local[(98)] =
            (SG2BMM_local[(98)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 130))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            138))]));
        SG2BMM_local[(101)] =
            (SG2BMM_local[(101)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 134))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            142))]));
        SG2BMM_local[(104)] =
            (SG2BMM_local[(104)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 138))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            146))]));
        SG2BMM_local[(107)] =
            (SG2BMM_local[(107)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 142))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            150))]));
        SG2BMM_local[(110)] =
            (SG2BMM_local[(110)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 146))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            154))]));
        SG2BMM_local[(113)] =
            (SG2BMM_local[(113)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 150))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            158))]));
        SG2BMM_local[(116)] =
            (SG2BMM_local[(116)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 154))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            162))]));
        SG2BMM_local[(119)] =
            (SG2BMM_local[(119)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 158))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            166))]));
        SG2BMM_local[(2)] =
            (SG2BMM_local[(2)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 3))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            11))]));
        SG2BMM_local[(5)] =
            (SG2BMM_local[(5)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 7))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            15))]));
        SG2BMM_local[(8)] =
            (SG2BMM_local[(8)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 11))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            19))]));
        SG2BMM_local[(11)] =
            (SG2BMM_local[(11)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 15))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            23))]));
        SG2BMM_local[(14)] =
            (SG2BMM_local[(14)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 19))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            27))]));
        SG2BMM_local[(17)] =
            (SG2BMM_local[(17)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 23))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            31))]));
        SG2BMM_local[(20)] =
            (SG2BMM_local[(20)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 27))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            35))]));
        SG2BMM_local[(23)] =
            (SG2BMM_local[(23)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 31))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            39))]));
        SG2BMM_local[(26)] =
            (SG2BMM_local[(26)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 35))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            43))]));
        SG2BMM_local[(29)] =
            (SG2BMM_local[(29)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 39))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            47))]));
        SG2BMM_local[(32)] =
            (SG2BMM_local[(32)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 43))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            51))]));
        SG2BMM_local[(35)] =
            (SG2BMM_local[(35)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 47))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            55))]));
        SG2BMM_local[(38)] =
            (SG2BMM_local[(38)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 51))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            59))]));
        SG2BMM_local[(41)] =
            (SG2BMM_local[(41)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 55))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            63))]));
        SG2BMM_local[(44)] =
            (SG2BMM_local[(44)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 59))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            67))]));
        SG2BMM_local[(47)] =
            (SG2BMM_local[(47)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 63))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            71))]));
        SG2BMM_local[(50)] =
            (SG2BMM_local[(50)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 67))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            75))]));
        SG2BMM_local[(53)] =
            (SG2BMM_local[(53)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 71))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            79))]));
        SG2BMM_local[(56)] =
            (SG2BMM_local[(56)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 75))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            83))]));
        SG2BMM_local[(59)] =
            (SG2BMM_local[(59)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 79))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            87))]));
        SG2BMM_local[(62)] =
            (SG2BMM_local[(62)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 83))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            91))]));
        SG2BMM_local[(65)] =
            (SG2BMM_local[(65)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 87))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            95))]));
        SG2BMM_local[(68)] =
            (SG2BMM_local[(68)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 91))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            99))]));
        SG2BMM_local[(71)] =
            (SG2BMM_local[(71)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 95))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            103))]));
        SG2BMM_local[(74)] =
            (SG2BMM_local[(74)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 99))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            107))]));
        SG2BMM_local[(77)] =
            (SG2BMM_local[(77)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 103))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            111))]));
        SG2BMM_local[(80)] =
            (SG2BMM_local[(80)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 107))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            115))]));
        SG2BMM_local[(83)] =
            (SG2BMM_local[(83)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 111))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            119))]));
        SG2BMM_local[(86)] =
            (SG2BMM_local[(86)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 115))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            123))]));
        SG2BMM_local[(89)] =
            (SG2BMM_local[(89)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 119))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            127))]));
        SG2BMM_local[(92)] =
            (SG2BMM_local[(92)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 123))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            131))]));
        SG2BMM_local[(95)] =
            (SG2BMM_local[(95)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 127))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            135))]));
        SG2BMM_local[(98)] =
            (SG2BMM_local[(98)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 131))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            139))]));
        SG2BMM_local[(101)] =
            (SG2BMM_local[(101)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 135))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            143))]));
        SG2BMM_local[(104)] =
            (SG2BMM_local[(104)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 139))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            147))]));
        SG2BMM_local[(107)] =
            (SG2BMM_local[(107)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 143))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            151))]));
        SG2BMM_local[(110)] =
            (SG2BMM_local[(110)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 147))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            155))]));
        SG2BMM_local[(113)] =
            (SG2BMM_local[(113)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 151))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            159))]));
        SG2BMM_local[(116)] =
            (SG2BMM_local[(116)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 155))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            163))]));
        SG2BMM_local[(119)] =
            (SG2BMM_local[(119)] +
             (q_shared[((((((int)threadIdx.x) / 29) * 160) + 159))] *
              Kpad_shared[(((((((int)threadIdx.x) / 29) * 160) +
                             ((((int)threadIdx.x) % 29) * 12)) +
                            167))]));
    }
    for (int j_inner = 0; j_inner < 40; ++j_inner) {
        for (int k_inner = 0; k_inner < 3; ++k_inner) {
            SG2BMM[((((((((((int)blockIdx.x) / 23) * 160080) +
                         ((((int)threadIdx.x) / 29) * 80040)) +
                        (j_inner * 2001)) +
                       ((((int)blockIdx.x) % 23) * 87)) +
                      ((((int)threadIdx.x) % 29) * 3)) +
                     k_inner))] = SG2BMM_local[(((j_inner * 3) + k_inner))];
        }
    }
}

inline __global__ void __launch_bounds__(276)
    sg2bmm_bs1_n10000_m64_w1000_d4_kernel0(float *__restrict__ q,
                                           float *__restrict__ k,
                                           float *__restrict__ SG2BMM) {
    float SG2BMM_local[145];
    __shared__ float q_shared[20];
    __shared__ float Kpad_shared[8020];
    for (int k_c_inner_init = 0; k_c_inner_init < 29; ++k_c_inner_init) {
        SG2BMM_local[(k_c_inner_init)] = 0.000000e+00f;
        SG2BMM_local[((k_c_inner_init + 29))] = 0.000000e+00f;
        SG2BMM_local[((k_c_inner_init + 58))] = 0.000000e+00f;
        SG2BMM_local[((k_c_inner_init + 87))] = 0.000000e+00f;
        SG2BMM_local[((k_c_inner_init + 116))] = 0.000000e+00f;
    }
    for (int p_outer_outer = 0; p_outer_outer < 64; ++p_outer_outer) {
        __syncthreads();
        if (((int)threadIdx.x) < 20) {
            q_shared[(((int)threadIdx.x))] =
                q[((((((int)blockIdx.x) * 1280) + (((int)threadIdx.x) * 64)) +
                    p_outer_outer))];
        }
        Kpad_shared[(((int)threadIdx.x))] =
            ((4000 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       256000))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 276))] =
            ((3724 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       238336))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 552))] =
            ((3448 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       220672))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 828))] =
            ((3172 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       203008))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 1104))] =
            ((2896 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       185344))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 1380))] =
            ((2620 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       167680))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 1656))] =
            ((2344 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       150016))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 1932))] =
            ((2068 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       132352))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 2208))] =
            ((1792 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       114688))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 2484))] =
            ((1516 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       97024))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 2760))] =
            ((1240 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       79360))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 3036))] =
            ((964 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       61696))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 3312))] =
            ((688 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       44032))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 3588))] =
            ((412 <= (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       26368))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 3864))] =
            (((136 <=
               (((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x))) &&
              ((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 10136))
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) -
                       8704))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 4140))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 9860)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       8960))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 4416))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 9584)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       26624))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 4692))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 9308)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       44288))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 4968))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 9032)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       61952))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 5244))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 8756)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       79616))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 5520))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 8480)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       97280))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 5796))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 8204)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       114944))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 6072))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 7928)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       132608))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 6348))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 7652)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       150272))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 6624))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 7376)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       167936))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 6900))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 7100)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       185600))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 7176))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 6824)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       203264))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 7452))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 6548)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       220928))]
                 : 0.000000e+00f);
        Kpad_shared[((((int)threadIdx.x) + 7728))] =
            (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) < 6272)
                 ? k[(((((((int)blockIdx.x) * 1280) +
                         (((int)threadIdx.x) * 64)) +
                        p_outer_outer) +
                       238592))]
                 : 0.000000e+00f);
        if (((int)threadIdx.x) < 16) {
            Kpad_shared[((((int)threadIdx.x) + 8004))] =
                (((((((int)blockIdx.x) % 500) * 20) + ((int)threadIdx.x)) <
                  5996)
                     ? k[(((((((int)blockIdx.x) * 1280) +
                             (((int)threadIdx.x) * 64)) +
                            p_outer_outer) +
                           256256))]
                     : 0.000000e+00f);
        }
        __syncthreads();
        for (int k_c_inner = 0; k_c_inner < 29; ++k_c_inner) {
            SG2BMM_local[(k_c_inner)] =
                (SG2BMM_local[(k_c_inner)] +
                 (q_shared[((((int)threadIdx.x) / 69))] *
                  Kpad_shared[(
                      ((((((int)threadIdx.x) % 69) * 116) + (k_c_inner * 4)) +
                       (((int)threadIdx.x) / 69)))]));
            SG2BMM_local[((k_c_inner + 29))] =
                (SG2BMM_local[((k_c_inner + 29))] +
                 (q_shared[(((((int)threadIdx.x) / 69) + 4))] *
                  Kpad_shared[(
                      (((((((int)threadIdx.x) % 69) * 116) + (k_c_inner * 4)) +
                        (((int)threadIdx.x) / 69)) +
                       4))]));
            SG2BMM_local[((k_c_inner + 58))] =
                (SG2BMM_local[((k_c_inner + 58))] +
                 (q_shared[(((((int)threadIdx.x) / 69) + 8))] *
                  Kpad_shared[(
                      (((((((int)threadIdx.x) % 69) * 116) + (k_c_inner * 4)) +
                        (((int)threadIdx.x) / 69)) +
                       8))]));
            SG2BMM_local[((k_c_inner + 87))] =
                (SG2BMM_local[((k_c_inner + 87))] +
                 (q_shared[(((((int)threadIdx.x) / 69) + 12))] *
                  Kpad_shared[(
                      (((((((int)threadIdx.x) % 69) * 116) + (k_c_inner * 4)) +
                        (((int)threadIdx.x) / 69)) +
                       12))]));
            SG2BMM_local[((k_c_inner + 116))] =
                (SG2BMM_local[((k_c_inner + 116))] +
                 (q_shared[(((((int)threadIdx.x) / 69) + 16))] *
                  Kpad_shared[(
                      (((((((int)threadIdx.x) % 69) * 116) + (k_c_inner * 4)) +
                        (((int)threadIdx.x) / 69)) +
                       16))]));
        }
    }
    for (int k_inner = 0; k_inner < 29; ++k_inner) {
        SG2BMM[((((((int)blockIdx.x) * 40020) + (((int)threadIdx.x) * 29)) +
                 k_inner))] = SG2BMM_local[(k_inner)];
        SG2BMM[(((((((int)blockIdx.x) * 40020) + (((int)threadIdx.x) * 29)) +
                  k_inner) +
                 8004))] = SG2BMM_local[((k_inner + 29))];
        SG2BMM[(((((((int)blockIdx.x) * 40020) + (((int)threadIdx.x) * 29)) +
                  k_inner) +
                 16008))] = SG2BMM_local[((k_inner + 58))];
        SG2BMM[(((((((int)blockIdx.x) * 40020) + (((int)threadIdx.x) * 29)) +
                  k_inner) +
                 24012))] = SG2BMM_local[((k_inner + 87))];
        SG2BMM[(((((((int)blockIdx.x) * 40020) + (((int)threadIdx.x) * 29)) +
                  k_inner) +
                 32016))] = SG2BMM_local[((k_inner + 116))];
    }
}

inline void sg2bmm(float *__restrict__ q, float *__restrict__ k,
                   float *__restrict__ y, int bs, int n, int m, int w, int d) {
    assert(bs % 8 == 0);
    assert(n == 10000);
    assert(m == 64);
    assert(w == 1000);
    // FIXME: Batch on the same place
    if (d == 1) {
        for (int i = 0; i < bs; i += 8) {
            sg2bmm_bs1_n10000_m64_w1000_d1_kernel0<<<23000, 58>>>(
                q + n * m, k + n * m,
                y + n * (2 * w + 1));
        }
    } else if (d == 4) {
        for (int i = 0; i < bs; i += 8) {
            sg2bmm_bs1_n10000_m64_w1000_d4_kernel0<<<4000, 276>>>(
                q + n * m, k + n * m, y + n * (2 * w + 1));
        }
    } else {
        assert(false);
    }
}

inline __global__ void __launch_bounds__(256)
    gbmml_bs1_n10000_m64_w1000_d1_kernel0(float *__restrict__ prob,
                                          float *__restrict__ q,
                                          float *__restrict__ G2BMM) {
    float G2BMM_local[8];
    __shared__ float prob_shared[736];
    __shared__ float Qpad_shared[4864];
    G2BMM_local[(0)] = 0.000000e+00f;
    G2BMM_local[(1)] = 0.000000e+00f;
    G2BMM_local[(2)] = 0.000000e+00f;
    G2BMM_local[(3)] = 0.000000e+00f;
    G2BMM_local[(4)] = 0.000000e+00f;
    G2BMM_local[(5)] = 0.000000e+00f;
    G2BMM_local[(6)] = 0.000000e+00f;
    G2BMM_local[(7)] = 0.000000e+00f;
    for (int p_outer_outer = 0; p_outer_outer < 87; ++p_outer_outer) {
        __syncthreads();
        prob_shared[(((int)threadIdx.x))] =
            prob[(((((((((int)blockIdx.x) / 625) * 40020000) +
                      ((((int)blockIdx.x) % 625) * 32016)) +
                     ((((int)threadIdx.x) / 23) * 2001)) +
                    (p_outer_outer * 23)) +
                   (((int)threadIdx.x) % 23)))];
        prob_shared[((((int)threadIdx.x) + 256))] =
            prob[((((((((((int)blockIdx.x) / 625) * 40020000) +
                       (((((int)threadIdx.x) + 256) / 368) * 20010000)) +
                      ((((int)blockIdx.x) % 625) * 32016)) +
                     ((((((int)threadIdx.x) + 256) % 368) / 23) * 2001)) +
                    (p_outer_outer * 23)) +
                   ((((int)threadIdx.x) + 3) % 23)))];
        if (((int)threadIdx.x) < 224) {
            prob_shared[((((int)threadIdx.x) + 512))] =
                prob[((((((((((int)blockIdx.x) / 625) * 40020000) +
                           (((((int)threadIdx.x) + 512) / 368) * 20010000)) +
                          ((((int)blockIdx.x) % 625) * 32016)) +
                         (((((int)threadIdx.x) + 144) / 23) * 2001)) +
                        (p_outer_outer * 23)) +
                       ((((int)threadIdx.x) + 6) % 23)))];
        }
        Qpad_shared[(((int)threadIdx.x))] =
            (((1000 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6)) < 11000))
                 ? q[(((((((((int)blockIdx.x) / 625) * 1280000) +
                          (p_outer_outer * 1472)) +
                         ((((int)blockIdx.x) % 625) * 1024)) +
                        ((int)threadIdx.x)) -
                       64000))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 256))] =
            (((996 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6)) < 10996))
                 ? q[(((((((((int)blockIdx.x) / 625) * 1280000) +
                          (p_outer_outer * 1472)) +
                         ((((int)blockIdx.x) % 625) * 1024)) +
                        ((int)threadIdx.x)) -
                       63744))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 512))] =
            (((992 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6)) < 10992))
                 ? q[(((((((((int)blockIdx.x) / 625) * 1280000) +
                          (p_outer_outer * 1472)) +
                         ((((int)blockIdx.x) % 625) * 1024)) +
                        ((int)threadIdx.x)) -
                       63488))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 768))] =
            (((988 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6)) < 10988))
                 ? q[(((((((((int)blockIdx.x) / 625) * 1280000) +
                          (p_outer_outer * 1472)) +
                         ((((int)blockIdx.x) % 625) * 1024)) +
                        ((int)threadIdx.x)) -
                       63232))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 1024))] =
            (((984 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6)) < 10984))
                 ? q[(((((((((int)blockIdx.x) / 625) * 1280000) +
                          (p_outer_outer * 1472)) +
                         ((((int)blockIdx.x) % 625) * 1024)) +
                        ((int)threadIdx.x)) -
                       62976))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 1280))] =
            (((980 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6)) < 10980))
                 ? q[(((((((((int)blockIdx.x) / 625) * 1280000) +
                          (p_outer_outer * 1472)) +
                         ((((int)blockIdx.x) % 625) * 1024)) +
                        ((int)threadIdx.x)) -
                       62720))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 1536))] =
            (((976 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6)) < 10976))
                 ? q[(((((((((int)blockIdx.x) / 625) * 1280000) +
                          (p_outer_outer * 1472)) +
                         ((((int)blockIdx.x) % 625) * 1024)) +
                        ((int)threadIdx.x)) -
                       62464))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 1792))] =
            (((972 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6)) < 10972))
                 ? q[(((((((((int)blockIdx.x) / 625) * 1280000) +
                          (p_outer_outer * 1472)) +
                         ((((int)blockIdx.x) % 625) * 1024)) +
                        ((int)threadIdx.x)) -
                       62208))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 2048))] =
            (((968 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((int)threadIdx.x) >> 6)) < 10968))
                 ? q[(((((((((int)blockIdx.x) / 625) * 1280000) +
                          (p_outer_outer * 1472)) +
                         ((((int)blockIdx.x) % 625) * 1024)) +
                        ((int)threadIdx.x)) -
                       61952))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 2304))] =
            (((1000 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((((int)threadIdx.x) >> 6) + 36) % 38))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                (((((int)threadIdx.x) >> 6) + 36) % 38)) < 11000))
                 ? q[(((((((((((int)blockIdx.x) / 625) * 1280000) +
                            (((((int)threadIdx.x) + 2304) / 2432) * 640000)) +
                           (p_outer_outer * 1472)) +
                          ((((int)blockIdx.x) % 625) * 1024)) +
                         ((((((int)threadIdx.x) >> 6) + 36) % 38) * 64)) +
                        (((int)threadIdx.x) & 63)) -
                       64000))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 2560))] =
            (((1000 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 2))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 2)) < 11000))
                 ? q[(((((((((((int)blockIdx.x) / 625) * 1280000) +
                            (((((int)threadIdx.x) + 2560) / 2432) * 640000)) +
                           (p_outer_outer * 1472)) +
                          ((((int)blockIdx.x) % 625) * 1024)) +
                         (((((int)threadIdx.x) >> 6) + 2) * 64)) +
                        (((int)threadIdx.x) & 63)) -
                       64000))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 2816))] =
            (((1000 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 6))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 6)) < 11000))
                 ? q[(((((((((((int)blockIdx.x) / 625) * 1280000) +
                            (((((int)threadIdx.x) + 2816) / 2432) * 640000)) +
                           (p_outer_outer * 1472)) +
                          ((((int)blockIdx.x) % 625) * 1024)) +
                         (((((int)threadIdx.x) >> 6) + 6) * 64)) +
                        (((int)threadIdx.x) & 63)) -
                       64000))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 3072))] =
            (((1000 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 10))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 10)) < 11000))
                 ? q[(((((((((((int)blockIdx.x) / 625) * 1280000) +
                            (((((int)threadIdx.x) + 3072) / 2432) * 640000)) +
                           (p_outer_outer * 1472)) +
                          ((((int)blockIdx.x) % 625) * 1024)) +
                         (((((int)threadIdx.x) >> 6) + 10) * 64)) +
                        (((int)threadIdx.x) & 63)) -
                       64000))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 3328))] =
            (((1000 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 14))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 14)) < 11000))
                 ? q[(((((((((((int)blockIdx.x) / 625) * 1280000) +
                            (((((int)threadIdx.x) + 3328) / 2432) * 640000)) +
                           (p_outer_outer * 1472)) +
                          ((((int)blockIdx.x) % 625) * 1024)) +
                         (((((int)threadIdx.x) >> 6) + 14) * 64)) +
                        (((int)threadIdx.x) & 63)) -
                       64000))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 3584))] =
            (((1000 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 18))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 18)) < 11000))
                 ? q[(((((((((((int)blockIdx.x) / 625) * 1280000) +
                            (((((int)threadIdx.x) + 3584) / 2432) * 640000)) +
                           (p_outer_outer * 1472)) +
                          ((((int)blockIdx.x) % 625) * 1024)) +
                         (((((int)threadIdx.x) >> 6) + 18) * 64)) +
                        (((int)threadIdx.x) & 63)) -
                       64000))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 3840))] =
            (((1000 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 22))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 22)) < 11000))
                 ? q[(((((((((((int)blockIdx.x) / 625) * 1280000) +
                            (((((int)threadIdx.x) + 3840) / 2432) * 640000)) +
                           (p_outer_outer * 1472)) +
                          ((((int)blockIdx.x) % 625) * 1024)) +
                         (((((int)threadIdx.x) >> 6) + 22) * 64)) +
                        (((int)threadIdx.x) & 63)) -
                       64000))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 4096))] =
            (((1000 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 26))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 26)) < 11000))
                 ? q[(((((((((((int)blockIdx.x) / 625) * 1280000) +
                            (((((int)threadIdx.x) + 4096) / 2432) * 640000)) +
                           (p_outer_outer * 1472)) +
                          ((((int)blockIdx.x) % 625) * 1024)) +
                         (((((int)threadIdx.x) >> 6) + 26) * 64)) +
                        (((int)threadIdx.x) & 63)) -
                       64000))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 4352))] =
            (((1000 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 30))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 30)) < 11000))
                 ? q[(((((((((((int)blockIdx.x) / 625) * 1280000) +
                            (((((int)threadIdx.x) + 4352) / 2432) * 640000)) +
                           (p_outer_outer * 1472)) +
                          ((((int)blockIdx.x) % 625) * 1024)) +
                         (((((int)threadIdx.x) >> 6) + 30) * 64)) +
                        (((int)threadIdx.x) & 63)) -
                       64000))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 4608))] =
            (((1000 <=
               (((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 34))) &&
              ((((p_outer_outer * 23) + ((((int)blockIdx.x) % 625) * 16)) +
                ((((int)threadIdx.x) >> 6) + 34)) < 11000))
                 ? q[(((((((((((int)blockIdx.x) / 625) * 1280000) +
                            (((((int)threadIdx.x) + 4608) / 2432) * 640000)) +
                           (p_outer_outer * 1472)) +
                          ((((int)blockIdx.x) % 625) * 1024)) +
                         (((((int)threadIdx.x) >> 6) + 34) * 64)) +
                        (((int)threadIdx.x) & 63)) -
                       64000))]
                 : 0.000000e+00f);
        __syncthreads();
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[(((((int)threadIdx.x) >> 5) * 92))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 7) * 2432) +
                             (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                            ((((int)threadIdx.x) & 31) * 2)))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[(((((int)threadIdx.x) >> 5) * 92))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 23))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            64))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 23))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            65))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 46))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            128))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 46))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            129))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 69))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            192))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 69))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            193))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 1))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            64))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 1))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            65))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 24))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            128))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 24))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            129))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 47))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            192))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 47))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            193))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 70))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            256))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 70))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            257))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 2))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            128))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 2))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            129))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 25))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            192))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 25))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            193))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 48))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            256))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 48))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            257))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 71))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            320))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 71))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            321))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 3))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            192))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 3))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            193))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 26))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            256))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 26))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            257))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 49))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            320))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 49))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            321))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 72))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            384))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 72))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            385))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 4))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            256))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 4))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            257))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 27))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            320))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 27))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            321))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 50))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            384))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 50))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            385))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 73))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            448))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 73))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            449))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 5))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            320))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 5))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            321))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 28))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            384))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 28))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            385))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 51))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            448))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 51))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            449))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 74))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            512))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 74))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            513))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 6))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            384))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 6))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            385))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 29))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            448))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 29))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            449))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 52))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            512))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 52))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            513))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 75))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            576))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 75))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            577))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 7))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            448))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 7))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            449))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 30))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            512))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 30))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            513))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 53))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            576))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 53))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            577))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 76))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            640))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 76))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            641))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 8))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            512))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 8))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            513))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 31))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            576))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 31))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            577))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 54))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            640))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 54))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            641))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 77))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            704))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 77))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            705))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 9))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            576))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 9))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            577))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 32))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            640))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 32))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            641))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 55))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            704))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 55))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            705))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 78))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            768))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 78))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            769))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 10))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            640))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 10))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            641))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 33))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            704))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 33))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            705))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 56))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            768))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 56))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            769))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 79))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            832))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 79))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            833))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 11))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            704))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 11))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            705))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 34))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            768))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 34))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            769))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 57))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            832))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 57))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            833))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 80))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            896))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 80))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            897))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 12))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            768))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 12))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            769))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 35))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            832))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 35))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            833))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 58))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            896))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 58))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            897))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 81))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            960))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 81))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            961))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 13))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            832))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 13))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            833))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 36))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            896))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 36))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            897))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 59))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            960))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 59))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            961))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 82))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1024))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 82))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1025))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 14))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            896))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 14))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            897))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 37))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            960))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 37))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            961))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 60))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1024))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 60))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1025))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 83))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1088))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 83))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1089))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 15))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            960))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 15))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            961))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 38))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1024))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 38))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1025))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 61))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1088))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 61))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1089))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 84))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1152))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 84))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1153))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 16))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1024))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 16))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1025))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 39))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1088))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 39))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1089))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 62))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1152))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 62))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1153))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 85))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1216))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 85))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1217))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 17))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1088))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 17))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1089))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 40))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1152))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 40))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1153))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 63))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1216))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 63))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1217))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 86))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1280))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 86))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1281))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 18))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1152))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 18))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1153))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 41))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1216))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 41))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1217))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 64))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1280))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 64))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1281))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 87))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1344))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 87))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1345))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 19))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1216))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 19))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1217))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 42))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1280))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 42))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1281))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 65))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1344))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 65))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1345))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 88))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1408))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 88))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1409))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 20))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1280))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 20))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1281))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 43))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1344))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 43))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1345))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 66))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1408))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 66))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1409))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 89))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1472))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 89))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1473))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 21))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1344))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 21))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1345))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 44))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1408))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 44))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1409))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 67))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1472))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 67))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1473))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 90))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1536))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 90))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1537))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 22))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1408))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 22))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1409))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 45))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1472))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 45))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1473))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 68))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1536))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 68))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1537))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 91))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1600))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 92) + 91))] *
              Qpad_shared[((((((((int)threadIdx.x) >> 7) * 2432) +
                              (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1601))]));
    }
    for (int j_inner = 0; j_inner < 4; ++j_inner) {
        for (int k_inner = 0; k_inner < 2; ++k_inner) {
            G2BMM[(((((((((((int)blockIdx.x) / 625) * 1280000) +
                         ((((int)threadIdx.x) >> 7) * 640000)) +
                        ((((int)blockIdx.x) % 625) * 1024)) +
                       (((((int)threadIdx.x) & 127) >> 5) * 256)) +
                      (j_inner * 64)) +
                     ((((int)threadIdx.x) & 31) * 2)) +
                    k_inner))] = G2BMM_local[(((j_inner * 2) + k_inner))];
        }
    }
}

inline __global__ void __launch_bounds__(320)
    gbmml_bs1_n10000_m64_w1000_d4_kernel0(float *__restrict__ prob,
                                          float *__restrict__ q,
                                          float *__restrict__ G2BMM) {
    float G2BMM_local[20];
    __shared__ float prob_shared[300];
    __shared__ float Qpad_shared[6912];
    G2BMM_local[(0)] = 0.000000e+00f;
    G2BMM_local[(2)] = 0.000000e+00f;
    G2BMM_local[(1)] = 0.000000e+00f;
    G2BMM_local[(3)] = 0.000000e+00f;
    G2BMM_local[(4)] = 0.000000e+00f;
    G2BMM_local[(6)] = 0.000000e+00f;
    G2BMM_local[(5)] = 0.000000e+00f;
    G2BMM_local[(7)] = 0.000000e+00f;
    G2BMM_local[(8)] = 0.000000e+00f;
    G2BMM_local[(10)] = 0.000000e+00f;
    G2BMM_local[(9)] = 0.000000e+00f;
    G2BMM_local[(11)] = 0.000000e+00f;
    G2BMM_local[(12)] = 0.000000e+00f;
    G2BMM_local[(14)] = 0.000000e+00f;
    G2BMM_local[(13)] = 0.000000e+00f;
    G2BMM_local[(15)] = 0.000000e+00f;
    G2BMM_local[(16)] = 0.000000e+00f;
    G2BMM_local[(18)] = 0.000000e+00f;
    G2BMM_local[(17)] = 0.000000e+00f;
    G2BMM_local[(19)] = 0.000000e+00f;
    for (int p_outer_outer = 0; p_outer_outer < 667; ++p_outer_outer) {
        __syncthreads();
        if (((int)threadIdx.x) < 100) {
            ((float3 *)(prob_shared + ((((int)threadIdx.x) * 3))))[0] =
                ((float3 *)(prob + ((((((int)blockIdx.x) * 200100) +
                                      (((int)threadIdx.x) * 2001)) +
                                     (p_outer_outer * 3)))))[0];
        }
        Qpad_shared[(((int)threadIdx.x))] =
            (((4000 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 14000))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       256000))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 320))] =
            (((3995 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13995))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       255680))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 640))] =
            (((3990 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13990))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       255360))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 960))] =
            (((3985 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13985))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       255040))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 1280))] =
            (((3980 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13980))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       254720))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 1600))] =
            (((3975 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13975))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       254400))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 1920))] =
            (((3970 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13970))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       254080))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 2240))] =
            (((3965 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13965))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       253760))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 2560))] =
            (((3960 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13960))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       253440))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 2880))] =
            (((3955 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13955))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       253120))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 3200))] =
            (((3950 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13950))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       252800))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 3520))] =
            (((3945 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13945))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       252480))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 3840))] =
            (((3940 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13940))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       252160))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 4160))] =
            (((3935 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13935))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       251840))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 4480))] =
            (((3930 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13930))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       251520))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 4800))] =
            (((3925 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13925))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       251200))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 5120))] =
            (((3920 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13920))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       250880))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 5440))] =
            (((3915 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13915))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       250560))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 5760))] =
            (((3910 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13910))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       250240))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 6080))] =
            (((3905 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13905))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       249920))]
                 : 0.000000e+00f);
        Qpad_shared[((((int)threadIdx.x) + 6400))] =
            (((3900 <=
               ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6))) &&
              (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                (((int)threadIdx.x) >> 6)) < 13900))
                 ? q[(((((((int)blockIdx.x) * 6400) + (p_outer_outer * 768)) +
                        ((int)threadIdx.x)) -
                       249600))]
                 : 0.000000e+00f);
        if (((int)threadIdx.x) < 192) {
            Qpad_shared[((((int)threadIdx.x) + 6720))] =
                (((3895 <=
                   ((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                    (((int)threadIdx.x) >> 6))) &&
                  (((((((int)blockIdx.x) % 100) * 100) + (p_outer_outer * 12)) +
                    (((int)threadIdx.x) >> 6)) < 13895))
                     ? q[(((((((int)blockIdx.x) * 6400) +
                             (p_outer_outer * 768)) +
                            ((int)threadIdx.x)) -
                           249280))]
                     : 0.000000e+00f);
        }
        __syncthreads();
        G2BMM_local[(0)] = (G2BMM_local[(0)] +
                            (prob_shared[(((((int)threadIdx.x) >> 5) * 30))] *
                             Qpad_shared[((((((int)threadIdx.x) >> 5) * 640) +
                                           ((((int)threadIdx.x) & 31) * 2)))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 3))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            64))]));
        G2BMM_local[(1)] = (G2BMM_local[(1)] +
                            (prob_shared[(((((int)threadIdx.x) >> 5) * 30))] *
                             Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                                            ((((int)threadIdx.x) & 31) * 2)) +
                                           1))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 3))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            65))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 6))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            128))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 9))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            192))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 6))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            129))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 9))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            193))]));
        G2BMM_local[(8)] =
            (G2BMM_local[(8)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 12))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            256))]));
        G2BMM_local[(10)] =
            (G2BMM_local[(10)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 15))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            320))]));
        G2BMM_local[(9)] =
            (G2BMM_local[(9)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 12))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            257))]));
        G2BMM_local[(11)] =
            (G2BMM_local[(11)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 15))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            321))]));
        G2BMM_local[(12)] =
            (G2BMM_local[(12)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 18))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            384))]));
        G2BMM_local[(14)] =
            (G2BMM_local[(14)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 21))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            448))]));
        G2BMM_local[(13)] =
            (G2BMM_local[(13)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 18))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            385))]));
        G2BMM_local[(15)] =
            (G2BMM_local[(15)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 21))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            449))]));
        G2BMM_local[(16)] =
            (G2BMM_local[(16)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 24))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            512))]));
        G2BMM_local[(18)] =
            (G2BMM_local[(18)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 27))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            576))]));
        G2BMM_local[(17)] =
            (G2BMM_local[(17)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 24))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            513))]));
        G2BMM_local[(19)] =
            (G2BMM_local[(19)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 27))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            577))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 1))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            256))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 4))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            320))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 1))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            257))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 4))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            321))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 7))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            384))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 10))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            448))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 7))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            385))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 10))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            449))]));
        G2BMM_local[(8)] =
            (G2BMM_local[(8)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 13))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            512))]));
        G2BMM_local[(10)] =
            (G2BMM_local[(10)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 16))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            576))]));
        G2BMM_local[(9)] =
            (G2BMM_local[(9)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 13))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            513))]));
        G2BMM_local[(11)] =
            (G2BMM_local[(11)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 16))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            577))]));
        G2BMM_local[(12)] =
            (G2BMM_local[(12)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 19))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            640))]));
        G2BMM_local[(14)] =
            (G2BMM_local[(14)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 22))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            704))]));
        G2BMM_local[(13)] =
            (G2BMM_local[(13)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 19))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            641))]));
        G2BMM_local[(15)] =
            (G2BMM_local[(15)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 22))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            705))]));
        G2BMM_local[(16)] =
            (G2BMM_local[(16)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 25))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            768))]));
        G2BMM_local[(18)] =
            (G2BMM_local[(18)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 28))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            832))]));
        G2BMM_local[(17)] =
            (G2BMM_local[(17)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 25))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            769))]));
        G2BMM_local[(19)] =
            (G2BMM_local[(19)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 28))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            833))]));
        G2BMM_local[(0)] =
            (G2BMM_local[(0)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 2))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            512))]));
        G2BMM_local[(2)] =
            (G2BMM_local[(2)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 5))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            576))]));
        G2BMM_local[(1)] =
            (G2BMM_local[(1)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 2))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            513))]));
        G2BMM_local[(3)] =
            (G2BMM_local[(3)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 5))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            577))]));
        G2BMM_local[(4)] =
            (G2BMM_local[(4)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 8))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            640))]));
        G2BMM_local[(6)] =
            (G2BMM_local[(6)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 11))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            704))]));
        G2BMM_local[(5)] =
            (G2BMM_local[(5)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 8))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            641))]));
        G2BMM_local[(7)] =
            (G2BMM_local[(7)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 11))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            705))]));
        G2BMM_local[(8)] =
            (G2BMM_local[(8)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 14))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            768))]));
        G2BMM_local[(10)] =
            (G2BMM_local[(10)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 17))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            832))]));
        G2BMM_local[(9)] =
            (G2BMM_local[(9)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 14))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            769))]));
        G2BMM_local[(11)] =
            (G2BMM_local[(11)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 17))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            833))]));
        G2BMM_local[(12)] =
            (G2BMM_local[(12)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 20))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            896))]));
        G2BMM_local[(14)] =
            (G2BMM_local[(14)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 23))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            960))]));
        G2BMM_local[(13)] =
            (G2BMM_local[(13)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 20))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            897))]));
        G2BMM_local[(15)] =
            (G2BMM_local[(15)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 23))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            961))]));
        G2BMM_local[(16)] =
            (G2BMM_local[(16)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 26))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1024))]));
        G2BMM_local[(18)] =
            (G2BMM_local[(18)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 29))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1088))]));
        G2BMM_local[(17)] =
            (G2BMM_local[(17)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 26))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1025))]));
        G2BMM_local[(19)] =
            (G2BMM_local[(19)] +
             (prob_shared[((((((int)threadIdx.x) >> 5) * 30) + 29))] *
              Qpad_shared[(((((((int)threadIdx.x) >> 5) * 640) +
                             ((((int)threadIdx.x) & 31) * 2)) +
                            1089))]));
    }
    for (int j_inner = 0; j_inner < 10; ++j_inner) {
        for (int k_inner = 0; k_inner < 2; ++k_inner) {
            G2BMM[((((((((int)blockIdx.x) * 6400) +
                       ((((int)threadIdx.x) >> 5) * 640)) +
                      (j_inner * 64)) +
                     ((((int)threadIdx.x) & 31) * 2)) +
                    k_inner))] = G2BMM_local[(((j_inner * 2) + k_inner))];
        }
    }
}

inline void sgbmml(float *__restrict__ q, float *__restrict__ k,
                   float *__restrict__ y, int bs, int n, int m, int w, int d) {
    assert(bs % 8 == 0);
    assert(n == 10000);
    assert(m == 64);
    assert(w == 1000);
    if (d == 1) {
        for (int i = 0; i < bs; i += 8) {
            // Hacks for OOM
            gbmml_bs1_n10000_m64_w1000_d1_kernel0<<<2500, 256>>>(
                q +  n * (2 * w + 1), k +  n * m,
                y +  n * m);
        }
    } else if (d == 4) {
        for (int i = 0; i < bs; i += 8) {
            // Hacks for OOM
            gbmml_bs1_n10000_m64_w1000_d4_kernel0<<<800, 320>>>(
                q + n * (2 * w + 1), k +  n * m,
                y + n * m);
        }
    } else {
        assert(false);
    }
}

} // namespace tpm

#endif // CUSTOM_OPS_CUH
