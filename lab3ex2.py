import math


def ex2():
    edible_e = (3 / 8) * math.log2(8 / 3) + (5 / 8) * math.log2(8 / 5)

    smooth_ho = (4 / 8) + (4 / 8 * ((1 / 4) * math.log2(4) + (3 / 4) * math.log2(4 / 3)))

    smooth_ig = edible_e - smooth_ho

    weight_ho = ((3 / 8) * ((1 / 3) * math.log2(3) + (2 / 3) * math.log2(3 / 2))) + (
            (5 / 8) * ((2 / 5) * math.log2(5 / 2) + (3 / 5) * math.log2(5 / 3)))
    weight_ig = edible_e - weight_ho

    print(edible_e)
    print(smooth_ho)
    print(smooth_ig)
    print(weight_ig)


ex2()
