import sys, multiprocessing
if __name__ == '__main__':
    print('Running main with argv:', sys.argv)
    with multiprocessing.Pool(1) as p:
        print('Pool created')
