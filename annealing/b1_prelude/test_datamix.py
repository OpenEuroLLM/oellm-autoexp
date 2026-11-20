from pathlib import Path
import tempfile,unittest
import sys
sys.path.insert(0,str(Path(__file__).parent/"datamix"))
import rebuild

class Mixtures(unittest.TestCase):
    def test_both_original_blends_reproduce_exact_jsc_bytes(self):
        for mix in ('dominant1','dominant2'):
            self.assertEqual(rebuild.render(mix),(Path(rebuild.__file__).parent/(mix+'.txt')).read_text())

if __name__=='__main__':unittest.main()
