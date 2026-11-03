"""Check that interrupted request batches resume through lm-eval's real cache."""
import argparse
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

from lm_eval.api.model import CachingLM,LM


class Interrupted(Exception):
    pass


class ChunkedLM(LM):
    def __init__(self,stop=None):
        super().__init__();self.stop=stop;self.calls=0
    def loglikelihood(self,requests):
        results=[]
        for request in requests:
            result=(-float(len(request.args[0])+len(request.args[1])),True)
            self.cache_hook.add_partial('loglikelihood',request.args,result)
            self.calls+=1;results.append(result)
            if self.calls==self.stop:raise Interrupted()
        return results
    def loglikelihood_rolling(self,requests):raise NotImplementedError()
    def generate_until(self,requests):raise NotImplementedError()


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    requests=[SimpleNamespace(args=(f'context-{i}',' continuation')) for i in range(8)]
    expected=ChunkedLM().loglikelihood(requests)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        db=str(Path(tmp)/'responses.db')
        first=ChunkedLM(stop=3);cached=CachingLM(first,db)
        try:cached.loglikelihood(requests)
        except Interrupted:pass
        else:raise AssertionError('interruption did not fire')
        cached.dbdict.commit();before=len(cached.dbdict);cached.dbdict.close()
        second=ChunkedLM();resumed=CachingLM(second,db)
        actual=resumed.loglikelihood(requests)
        assert before==3 and second.calls==5 and actual==expected
        assert len(resumed.dbdict)==8
        resumed.dbdict.close()
    result={'passed':True,'requests':8,'persisted_before_interruption':before,
            'executed_on_resume':second.calls,'outputs_match_uninterrupted':actual==expected}
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
