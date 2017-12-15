def myround(x, base=200):
    return int(base * round(float(x)/base))

def main():
    interval = 2944
    interval = 4586
    x = [interval*(i+1) for i in range(17)] 
    x = [myround(i) for i in x]
    print("the list of iterations are: {}".format(x))

# hip [3000, 5800, 8800, 11800, 14800, 17600, 20600, 23600, 26400, 29400, 32400, 35400, 38200, 41200, 44200, 47200, 50000]
# socket [4600, 9200, 13800, 18400, 23000, 27600, 32200, 36600, 41200, 45800,
# 50400, 55000, 59600, 64200, 68800]
main()
