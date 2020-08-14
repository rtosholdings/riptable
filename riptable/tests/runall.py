# $Id: //Depot/Source/SFW/riptable/Python/core/riptable/tests/runall.py#1 $

import unittest
import sys, os

def get_all_tests():
   """
   :return:
   """
   tests = (fname[:-3] for fname in os.listdir(os.path.dirname(__file__) or '.')
            #if (fname.startswith('test_datetime.py') or fname.startswith('test_categorical.py')) and fname.endswith('.py'))
            if fname.startswith('test_') and fname.endswith('.py'))
   return sorted(tests)

def run_all( argv, verbosity = 3 ):
   if len( argv ) > 1:
      try:
         verbosity = int(argv[1])
      except:
         raise SystemExit( "Usage: %s [ verbosity_int (def=%d) ]" % ( argv[ 0 ], verbosity ) )
   pkg_name = os.path.basename(os.path.dirname(__file__))
   failures = []
   for test in get_all_tests():
      print("==> Running tests: %s <==\n" % test, file=sys.stderr)
      module_name = 'riptable.{pkg_name}.{mod_name}'.format(pkg_name=pkg_name, mod_name=test)
      module = __import__(module_name, fromlist=[''])
      utest = unittest.main(module=module, exit=False, argv=[module_name], verbosity=verbosity).result
      if not utest.wasSuccessful():
         failures.append(module_name)
   if failures:
      raise SystemExit('Failure: {}'.format(', '.join(failures)))


if __name__ == "__main__":
   run_all( sys.argv )
