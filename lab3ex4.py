import math


def ex4():
    # 2)
    c_e = (6 / 10) * math.log2(10 / 6) + (4 / 10) * math.log2(10 / 4)
    t_e = ((35 / 100) * math.log2(100 / 35)) * 2 + ((3 / 10) * math.log2(10 / 3))

    print(c_e)
    print(t_e)

    # 3):
    hc_t = ((1 / 3) * ((35 / 100) * math.log2(100 / 35) + (3 / 10) * math.log2(10 / 3)) * 2) + (1 / 3) * (
                2 * ((35 / 100) * math.log2(100 / 35)))
    ht_c = (1 / 2) * ((4 / 10) * math.log2(10 / 4)) + (1 / 2) * ((6 / 10) * math.log2(10 / 6))

    print(hc_t)
    print(ht_c)


ex4()
