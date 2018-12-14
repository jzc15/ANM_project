if __name__ == '__main__':
    inputfile = open('r.csv', 'r')
    k = 0
    for line in inputfile:
        k += 1
        if line.startswith('K'):
            continue
        x = line.strip().split(',')
        
        if x[2] not in ['0', '1']:
            print(k, x[2])

    assert k == 2345212