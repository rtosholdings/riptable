import pandas as pd
import riptable as rt
import random as rand


class MultiKeyGroupBy_Test:
    ##data generation code
    def test_multkey(self):
        alpha = 'Q W E R T Y U I O P A S D F G H J K L Z X C V B N M'.split(' ')
        digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

        sz = 4000
        numbers = [0] * sz
        keys1 = [''] * sz
        keys2 = [''] * sz

        for i in range(0, sz):
            numbers[i] = digits[rand.randint(0, 1000) % len(digits)]
            keys1[i] = alpha[rand.randint(0, 1000) % len(alpha)]
            keys2[i] = alpha[rand.randint(0, 1000) % len(alpha)]

        ary = rt.FastArray(numbers)

        data = {'k1': keys1, 'k2': keys2, 'beta': numbers}

        # print('SFW--------------------------------------------------------------')
        mset = rt.Dataset(data)

        # t = time.time()
        s_group = rt.GroupBy(mset, keys=['k1', 'k2']).sum()
        # print(time.time() - t, 'SFW GROUP BY ')

        # print('PANDAS--------------------------------------------------------------')
        df2 = pd.DataFrame(data)

        # t = time.time()
        p_group = df2.groupby(['k1', 'k2']).sum()
        # print(time.time() - t, 'PANDAS GROUP BY ')
        # print('compare out--------------------------------------------------------------')

        pandas = list(p_group['beta'])
        sfw = list(s_group['beta'])
        assert pandas == sfw

    def test_advanced_multikey(self):
        ##data generation code
        alpha = 'Q W E R T Y U I O P A S D F G H J K L Z X C V B N M'.split(' ')
        digits = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

        sz = 200
        numb_kvs = 5  # can't be more than 26 as we are usign the contents of alpha for the column name

        # 2d array of keys/values
        vals = [[0] * sz] * numb_kvs
        keys = [[''] * sz] * numb_kvs

        # random initialization for them
        for n in range(0, numb_kvs):
            for i in range(0, sz):
                vals[n][i] = digits[rand.randint(0, 1000) % len(digits)]
                keys[n][i] = alpha[rand.randint(0, 1000) % len(alpha)]

        # create the data map
        # multi key hash for numbkeys 1:numb_kvs
        while numb_kvs > 0:
            data = {}
            for n in range(0, numb_kvs):
                data[alpha[n]] = keys[n]
                data[alpha[n + numb_kvs]] = vals[n]

            key_cols = alpha[0:numb_kvs]
            val_cols = alpha[numb_kvs : numb_kvs * 2]

            # print('SFW--------------------------------------------------------------')
            mset = rt.Dataset(data)
            # t = time.time()
            s_group = rt.GroupBy(mset, keys=key_cols).sum()
            # print(time.time() - t, 'SFW GROUP BY ')

            # print('PANDAS--------------------------------------------------------------')
            df2 = pd.DataFrame(data)
            # t = time.time()
            p_group = df2.groupby(key_cols).sum()
            # print(time.time() - t, 'PANDAS GROUP BY ')
            # print('compare out--------------------------------------------------------------')

            pandas_ = list(p_group[val_cols])
            sfw_ = list(s_group[val_cols])

            assert pandas_ == sfw_

            numb_kvs = numb_kvs - 1


MultiKeyGroupBy_Test().test_advanced_multikey()
MultiKeyGroupBy_Test().test_multkey()
