"""CPU integration checks with real Git objects and local bare remotes."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('release',Path(__file__).parents[1]/'tools/megatron_release.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.auto=self.root/'auto';self.mega=self.root/'mega'
        for path in (self.auto,self.mega):
            subprocess.run(['git','init','-q',str(path)],check=True)
            r.git(path,'config','user.name','Release test')
            r.git(path,'config','user.email','test@example.invalid')
            (path/'README.md').write_text('source\n')
            r.git(path,'add','README.md');r.git(path,'commit','-qm','Initial')
        self.mega_remote=self.root/'mega.git';self.auto_remote=self.root/'auto.git'
        for path in (self.mega_remote,self.auto_remote):
            subprocess.run(['git','init','--bare','-q',str(path)],check=True)
        (self.auto/'.gitmodules').write_text(f'[submodule "{r.SUBMODULE}"]\npath = {r.SUBMODULE}\nurl = {self.mega_remote}\n')
        r.git(self.auto,'add','.gitmodules')
        r.git(self.auto,'update-index','--add','--cacheinfo',f'160000,{r.git(self.mega,"rev-parse","HEAD")},{r.SUBMODULE}')
        r.git(self.auto,'commit','-qm','Pin dependency')
        self.args=SimpleNamespace(autoexp=self.auto,megatron=self.mega,output=self.root/'package',
            name='test-v1',autoexp_branch='release/test',megatron_branch='release/test',
            autoexp_remote=str(self.auto_remote),megatron_remote=str(self.mega_remote),
            evidence='README.md',execute=True,assisted_by=None)
        # Production CLI admits canonical GitHub HTTPS only. Tests use isolated
        # bare repos; no network or credentials are involved.
        self.urls=patch.object(r,'url',side_effect=lambda value:value);self.urls.start();self.addCleanup(self.urls.stop)
        self.env=patch.dict('os.environ',{'GIT_AUTHOR_NAME':'Release test','GIT_AUTHOR_EMAIL':'test@example.invalid',
            'GIT_COMMITTER_NAME':'Release test','GIT_COMMITTER_EMAIL':'test@example.invalid',
            'GIT_CONFIG_COUNT':'1','GIT_CONFIG_KEY_0':'protocol.file.allow','GIT_CONFIG_VALUE_0':'always'})
        self.env.start();self.addCleanup(self.env.stop)

    def test_prepare_preserves_sources_and_pins_exact_tree(self):
        before=[(r.identity(p,True),r.git(p,'show-ref')) for p in (self.auto,self.mega)]
        self.args.execute=False;r.prepare(self.args);self.assertFalse(self.args.output.exists())
        self.args.execute=True;out=r.prepare(self.args)
        self.assertEqual(out['megatron'],r.identity(self.mega))
        self.assertEqual(before,[(r.identity(p,True),r.git(p,'show-ref')) for p in (self.auto,self.mega)])
        r.verify_package(self.args.output)
        with self.assertRaises(FileExistsError):r.prepare(self.args)

    def test_dirty_hidden_or_untracked_sources_rejected(self):
        (self.mega/'README.md').write_text('dirty')
        with self.assertRaises(ValueError):r.plan(self.args)
        r.git(self.mega,'checkout','--','README.md')
        r.git(self.mega,'update-index','--assume-unchanged','README.md')
        with self.assertRaises(ValueError):r.plan(self.args)
        r.git(self.mega,'update-index','--no-assume-unchanged','README.md')
        (self.mega/'new.py').write_text('uncommitted')
        with self.assertRaises(ValueError):r.plan(self.args)

    def test_modified_release_rejected(self):
        r.prepare(self.args)
        (self.args.output/'oellm-autoexp'/r.SUBMODULE/'README.md').write_text('drift')
        with self.assertRaises(ValueError):r.publish(self.args.output,True)
        self.assertEqual(r.git(self.auto_remote,'for-each-ref'),'')

    def test_publish_dependency_first_and_remote_round_trip(self):
        out=r.prepare(self.args)
        r.publish(self.args.output,False)
        self.assertEqual(r.git(self.auto_remote,'for-each-ref'),'')
        r.publish(self.args.output,True)
        r.publish(self.args.output,True)  # Retry is idempotent, no new commits/tags.
        result=r.verify_remote(self.args.output)
        self.assertTrue(result['fresh_remote_clone'])
        self.assertEqual(result['autoexp'],out['autoexp'])

    def test_dependency_failure_never_publishes_autoexp(self):
        r.prepare(self.args)
        with patch.object(r,'publish_repo',side_effect=RuntimeError('dependency failure')) as publish:
            with self.assertRaises(RuntimeError):r.publish(self.args.output,True)
        self.assertEqual(publish.call_count,1)
        self.assertEqual(r.git(self.auto_remote,'for-each-ref'),'')

    def test_existing_different_tag_never_overwritten(self):
        r.prepare(self.args);r.publish(self.args.output,True)
        other=self.root/'other';r.clone(self.mega,other,r.identity(self.mega)['commit'])
        r.git(other,'tag','-a','test-v1','-m','different annotation')
        with self.assertRaises(ValueError):
            r.publish_repo(other,str(self.mega_remote),'release/test','test-v1',
                r.identity(other)['commit'],r.git(other,'rev-parse','test-v1'),True)

    def test_diverged_branch_never_overwritten(self):
        r.prepare(self.args);r.publish(self.args.output,True)
        other=self.root/'other';r.clone(self.mega,other,r.identity(self.mega)['commit'])
        (other/'README.md').write_text('independent newer work')
        r.git(other,'add','README.md');r.git(other,'commit','-qm','Other owner')
        r.git(other,'push',str(self.mega_remote),'HEAD:refs/heads/release/test')
        with self.assertRaises(subprocess.CalledProcessError):r.publish(self.args.output,True)
        self.assertEqual(r.git(self.mega_remote,'rev-parse','refs/heads/release/test'),r.git(other,'rev-parse','HEAD'))


class UrlTests(unittest.TestCase):
    def test_no_secret_or_local_manifest_urls(self):
        self.assertEqual(r.url('https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM.git'),
                         'https://github.com/OpenEuroLLM/NVIDIA-Megatron-LM.git')
        for value in ('https://secret@github.com/a/b.git','https://github.com/a/b.git?token=x','/tmp/source'):
            with self.assertRaises(ValueError):r.url(value)


if __name__=='__main__':unittest.main()
