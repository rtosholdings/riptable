import unittest
import riptable as rt
import numpy as np
from io import StringIO
import sys


class Meta_Test(unittest.TestCase):
    def test_meta(self):
        st = rt.Struct(
            {
                'a': rt.Dataset(
                    {
                        'col1': rt.FastArray([1, 2]).astype(np.int32),
                        'col2': rt.FastArray([3, 4]).astype(np.int32),
                        'col4': rt.FastArray([5, 6]).astype(np.int32),
                    }
                ),
                'b': rt.FastArray([3, 4]).astype(np.int32),
            }
        )
        out = StringIO()
        orig_stdout = sys.stdout
        sys.stdout = out
        print(st.info())
        output = out.getvalue()
        target_output = '''\x1b[1;36mDescription: \x1b[00m<no description>
\x1b[1;36mSteward: \x1b[00m<no steward>
\x1b[1;36mType: \x1b[00mStruct
\x1b[1;36mContents:\x1b[00m

\x1b[1;36mType     Name  Description                                         Steward     \x1b[00m
\x1b[1;36m-------  ----  --------------------------------------------------  ------------\x1b[00m
Dataset  \x1b[1;32ma   \x1b[00m  <no description>                                    <no steward>
int32    \x1b[1;32mb   \x1b[00m  <no description>                                    <no steward>
'''
        self.assertEqual(output, target_output)

        schema = {'Description': 'This is a structure', 'Steward': 'Nick'}
        st.apply_schema(schema)
        st2 = rt.Struct({'This': st, 'That': np.array([1, 2]).astype(np.int32)})
        out = StringIO()
        sys.stdout = out
        print(st2.info())
        output = out.getvalue()
        target_output = '''\x1b[1;36mDescription: \x1b[00m<no description>
\x1b[1;36mSteward: \x1b[00m<no steward>
\x1b[1;36mType: \x1b[00mStruct
\x1b[1;36mContents:\x1b[00m

\x1b[1;36mType    Name  Description                                         Steward     \x1b[00m
\x1b[1;36m------  ----  --------------------------------------------------  ------------\x1b[00m
Struct  \x1b[1;32mThis\x1b[00m  This is a structure                                 Nick        
int32   \x1b[1;32mThat\x1b[00m  <no description>                                    <no steward>
'''
        self.assertEqual(output, target_output)

        schema = {
            'Description': 'This is a structure',
            'Steward': 'Nick',
            'Type': 'Struct',
            'Contents': {
                'This': {
                    'Description': 'This is a nested structure',
                    'Steward': 'Bob',
                    'Type': 'AttackHelicoptor',
                    'Contents': {
                        'a': {
                            'Description': 'A description for a',
                            'Steward': 'Fred',
                            'Contents': {
                                'col1': {
                                    'Description': 'This describes column 1',
                                    'Steward': 'Jay',
                                    'Type': 'int32',
                                },
                                'col2': {
                                    'Description': 'This describes column 2',
                                    'Steward': 'Alex',
                                    'Type': 'float32',
                                },
                                'col3': {
                                    'Description': 'This column is not there',
                                    'Steward': 'Ben',
                                },
                            },
                        },
                        'b': {
                            'Description': 'A descriptiion for b',
                            'Steward': 'George',
                        },
                    },
                },
                'That': {'Description': 'This is an array', 'Steward': 'Willy'},
            },
        }

        res = st2.apply_schema(schema)
        res_c = {
            'This': {
                'Type Mismatch': 'Type Struct does not match schema type AttackHelicoptor',
                'a': {
                    'col2': {
                        'Type Mismatch': 'Type int32 does not match schema type float32'
                    },
                    'Extra Column': 'col4',
                    'Missing Column': 'col3',
                },
            }
        }
        self.assertEqual(res, res_c)

        out = StringIO()
        sys.stdout = out
        print(st2.info())
        output = out.getvalue()
        target_output = '''\x1b[1;36mDescription: \x1b[00mThis is a structure
\x1b[1;36mSteward: \x1b[00mNick
\x1b[1;36mType: \x1b[00mStruct
\x1b[1;36mContents:\x1b[00m

\x1b[1;36mType    Name  Description                                         Steward     \x1b[00m
\x1b[1;36m------  ----  --------------------------------------------------  ------------\x1b[00m
Struct  \x1b[1;32mThis\x1b[00m  This is a nested structure                          Bob         
int32   \x1b[1;32mThat\x1b[00m  This is an array                                    Willy       
'''
        self.assertEqual(output, target_output)

        out = StringIO()
        sys.stdout = out
        print(st2.This.info())
        output = out.getvalue()
        target_output = '''\x1b[1;36mDescription: \x1b[00mThis is a nested structure
\x1b[1;36mSteward: \x1b[00mBob
\x1b[1;36mType: \x1b[00mStruct
\x1b[1;36mContents:\x1b[00m

\x1b[1;36mType     Name  Description                                         Steward     \x1b[00m
\x1b[1;36m-------  ----  --------------------------------------------------  ------------\x1b[00m
Dataset  \x1b[1;32ma   \x1b[00m  A description for a                                 Fred        
int32    \x1b[1;32mb   \x1b[00m  A descriptiion for b                                George      
'''
        self.assertEqual(output, target_output)

        out = StringIO()
        sys.stdout = out
        print(st2.This.a.info())
        output = out.getvalue()
        target_output = '''\x1b[1;36mDescription: \x1b[00mA description for a
\x1b[1;36mSteward: \x1b[00mFred
\x1b[1;36mType: \x1b[00mDataset
\x1b[1;36mContents:\x1b[00m

\x1b[1;36mType   Name  Description                                         Steward     \x1b[00m
\x1b[1;36m-----  ----  --------------------------------------------------  ------------\x1b[00m
int32  \x1b[1;32mcol1\x1b[00m  This describes column 1                             Jay         
int32  \x1b[1;32mcol2\x1b[00m  This describes column 2                             Alex        
int32  \x1b[1;32mcol4\x1b[00m  <no description>                                    <no steward>
'''
        self.assertEqual(output, target_output)

        out = StringIO()
        sys.stdout = out
        print(st2.This.a.col1.info())
        output = out.getvalue()
        target_output = '''\x1b[1;36mDescription: \x1b[00mThis describes column 1
\x1b[1;36mSteward: \x1b[00mJay
\x1b[1;36mType: \x1b[00mint32
'''
        self.assertEqual(output, target_output)

        sys.stdout = orig_stdout


if __name__ == "__main__":
    tester = unittest.main()
