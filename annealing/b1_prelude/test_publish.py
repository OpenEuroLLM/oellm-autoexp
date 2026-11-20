import tempfile,unittest
import types,sys
from unittest.mock import patch
from pathlib import Path
import publish
from test_validate import make_source_repo, git_at, validation

class Publication(unittest.TestCase):
    def test_reviewed_source_change_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo=Path(tmp);f=repo/'annealing/b1_prelude/a.py';f.parent.mkdir(parents=True);f.write_text('changed')
            with self.assertRaisesRegex(ValueError,'source changed'):
                publish.verify_files(dict(sources={'annealing/b1_prelude/a.py':'wrong'}),repo)

    def test_reviewed_path_must_stay_inside_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'package scope'):
                publish.verify_files(dict(sources={'../escape':'x'}),Path(tmp))

    def test_missing_training_validation_never_calls_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);publish.save(root/'validation.json',{'pending':True})
            publish.save(root/'publish-spec.json',dict(sources={},validation_manifest_sha256=publish.sha(root/'validation.json')))
            fake=types.SimpleNamespace(verify=lambda root: (_ for _ in ()).throw(ValueError('Training not completed')))
            with patch.dict(sys.modules,{'validate':fake}),patch.object(publish,'git') as git:
                with self.assertRaisesRegex(ValueError,'Training not completed'):publish.cycle(root,execute=True)
                git.assert_not_called()


class PublicationGit(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.repo=self.base/'repo';package=self.repo/'annealing/b1_prelude'
        package.mkdir(parents=True)
        (package/'a.py').write_text('# reviewed package\n')
        (package/'origins.json').write_text('{"origin": "reviewed"}\n')
        model=self.repo/'config/experiments/oellm_32b_dense/b1_prelude_annealing.yaml'
        model.parent.mkdir(parents=True);model.write_text('model: reviewed\n')
        self.baseline=make_source_repo(self.repo)
        git_at(self.repo,'config','user.name','Publication test')
        git_at(self.repo,'config','user.email','test@example.invalid')
        self.remote=self.base/'remote.git'
        import subprocess
        subprocess.run(['git','init','--bare','-q',str(self.remote)],check=True)
        git_at(self.repo,'push',str(self.remote),'HEAD:refs/heads/prod/test')
        self.root=self.base/'state';self.root.mkdir()
        paths=['annealing/b1_prelude/a.py','annealing/b1_prelude/origins.json','config/experiments/oellm_32b_dense/b1_prelude_annealing.yaml']
        identity=validation.capture_source_tree(self.repo,paths) if hasattr(validation,'capture_source_tree') else self.baseline
        publish.save(self.root/'validation.json',dict(source_tree=identity,arms=[]))
        publish.save(self.root/'publish-spec.json',dict(sources={p:publish.sha(self.repo/p) for p in paths},base=self.baseline['base_head'],remote_head=self.baseline['base_head'],validation_manifest_sha256=publish.sha(self.root/'validation.json')))
        self.evidence=dict(state='all_passed',manifest_sha256=publish.sha(self.root/'validation.json'),arms=[])
        fake=types.SimpleNamespace(verify=lambda root:self.evidence,check_source_tree=validation.check_source_tree)
        self.fake=fake
        for manager in [patch.object(publish,'REPO',self.repo),patch.object(publish,'REMOTE',str(self.remote)),patch.object(publish,'BRANCH','prod/test'),patch.dict(sys.modules,{'validate':fake})]:
            manager.start();self.addCleanup(manager.stop)
        original=publish.verify_files
        manager=patch.object(publish,'verify_files',side_effect=lambda spec,*a,**k:original(spec,self.repo))
        manager.start();self.addCleanup(manager.stop)

    def head(self):return git_at(self.repo,'rev-parse','HEAD')
    def remote_head(self):return git_at(self.remote,'rev-parse','refs/heads/prod/test')

    def test_stop_created_during_validation_prevents_all_git_writes(self):
        def checked(root):
            (root/'STOP').touch();return self.evidence
        self.fake.verify=checked
        with self.assertRaisesRegex(ValueError,'STOP'):publish.cycle(self.root,execute=True)
        self.assertEqual(self.head(),self.baseline['base_head'])
        self.assertEqual(git_at(self.repo,'diff','--cached','--name-only'),'')
        self.assertFalse((self.repo/'annealing/b1_prelude/validation_result.json').exists())

    def test_stop_after_local_commit_prevents_push(self):
        original=publish.git
        def interrupted(*args,**kwargs):
            result=original(*args,**kwargs)
            if args[0]=='commit':(self.root/'STOP').touch()
            return result
        with patch.object(publish,'git',side_effect=interrupted):
            with self.assertRaisesRegex(ValueError,'STOP'):publish.cycle(self.root,execute=True)
        self.assertNotEqual(self.head(),self.baseline['base_head'])
        self.assertEqual(self.remote_head(),self.baseline['base_head'])

    def test_commit_receipt_crash_reconciles_one_exact_commit(self):
        original=publish.save
        def crash(path,value):
            if Path(path).name=='publication.json':raise OSError('crash after commit')
            return original(path,value)
        with patch.object(publish,'save',side_effect=crash):
            with self.assertRaisesRegex(OSError,'crash after commit'):publish.cycle(self.root,execute=True)
        candidate=self.head()
        self.assertNotEqual(candidate,self.baseline['base_head'])
        result=publish.cycle(self.root,execute=True)
        self.assertEqual(result['published_commit'],candidate)
        self.assertEqual(self.head(),candidate)
        self.assertEqual(self.remote_head(),candidate)
        self.assertEqual(git_at(self.repo,'rev-list','--count',self.baseline['base_head']+'..HEAD'),'1')

    def test_push_receipt_crash_reconstructs_receipt_without_second_push(self):
        original_save=publish.save;original_git=publish.git;pushes=[]
        def crash(path,value):
            if Path(path).name=='publication.json' and value.get('published_commit'):raise OSError('crash after push')
            return original_save(path,value)
        def counted(*args,**kwargs):
            if 'push' in args:pushes.append(args)
            return original_git(*args,**kwargs)
        with patch.object(publish,'save',side_effect=crash),patch.object(publish,'git',side_effect=counted):
            with self.assertRaisesRegex(OSError,'crash after push'):publish.cycle(self.root,execute=True)
        candidate=self.head();self.assertEqual(self.remote_head(),candidate)
        with patch.object(publish,'git',side_effect=counted):result=publish.cycle(self.root,execute=True)
        self.assertEqual(result['published_commit'],candidate)
        self.assertEqual(len(pushes),1)
        self.assertFalse(any('--force' in arg or arg=='-f' for arg in pushes[0]))

    def test_precommit_intent_can_resume_only_its_owned_staged_files(self):
        original=publish.git
        def crash(*args,**kwargs):
            if args[0]=='commit':raise OSError('crash before commit')
            return original(*args,**kwargs)
        with patch.object(publish,'git',side_effect=crash):
            with self.assertRaisesRegex(OSError,'crash before commit'):publish.cycle(self.root,execute=True)
        intent=publish.read(self.root/'publication-intent.json')
        self.assertIn('result_payload',intent)
        self.assertIn('expected_files',intent)
        self.assertEqual(self.head(),self.baseline['base_head'])
        result=publish.cycle(self.root,execute=True)
        self.assertEqual(result['published_commit'],self.head())
        self.assertEqual(git_at(self.repo,'rev-list','--count',self.baseline['base_head']+'..HEAD'),'1')

    def test_existing_other_owner_index_is_preserved_without_result_write(self):
        (self.repo/'templates/production.sbatch').write_text('other owner\n')
        git_at(self.repo,'add','--','templates/production.sbatch')
        with self.assertRaises(ValueError):publish.cycle(self.root,execute=True)
        self.assertEqual(git_at(self.repo,'show',':templates/production.sbatch'),'other owner')
        self.assertFalse((self.repo/'annealing/b1_prelude/validation_result.json').exists())

    def test_new_staged_work_before_add_is_preserved(self):
        original=publish.write_guard;injected=[]
        source=self.repo/'annealing/b1_prelude/a.py';expected=source.read_text()
        def raced(root,spec):
            result=original(root,spec)
            if (self.repo/'annealing/b1_prelude/validation_result.json').exists() and not injected:
                source.write_text('another owner staged this\n');git_at(self.repo,'add','--','annealing/b1_prelude/a.py');source.write_text(expected);injected.append(True)
            return result
        with patch.object(publish,'write_guard',side_effect=raced):
            with self.assertRaises(ValueError):publish.cycle(self.root,execute=True)
        self.assertEqual(git_at(self.repo,'show',':annealing/b1_prelude/a.py'),'another owner staged this')
        self.assertEqual(self.head(),self.baseline['base_head'])

    def add_reports(self):
        arms=[]
        for i,name in enumerate(['dominant1','dominant2']):
            path=self.root/(name+'-report.json')
            report=dict(job=str(201+i),manifest_sha256=self.evidence['manifest_sha256'],restored_iteration=416000,
                steps=[dict(iteration=416001,lm_loss=1.2,iteration_ms=5000,throughput_tflops_per_gpu=100),dict(iteration=416002,lm_loss=1.1,loss_delta=-0.1),dict(iteration=416003,lm_loss=1.15,loss_delta=0.05)],scheduler=dict(state='COMPLETED',gpu_hours=10))
            publish.save(path,report)
            arms.append(dict(name=name,state='passed',job=report['job'],report=str(path),report_sha256=publish.sha(path)))
        self.evidence['arms']=arms

    def test_published_result_embeds_checked_per_arm_metric_reports(self):
        self.add_reports()
        publish.cycle(self.root,execute=True)
        result=publish.read(self.repo/'annealing/b1_prelude/validation_result.json')
        self.assertEqual(result['arm_reports']['dominant1']['restored_iteration'],416000)
        self.assertEqual(result['arm_reports']['dominant2']['steps'][2]['lm_loss'],1.15)
        self.assertEqual(result['arm_reports']['dominant1']['scheduler']['gpu_hours'],10)

    def test_changed_metric_report_is_rejected_before_writing_result(self):
        self.add_reports()
        Path(self.evidence['arms'][0]['report']).write_text('{"changed":true}')
        with self.assertRaises(ValueError):publish.cycle(self.root,execute=True)
        self.assertEqual(self.head(),self.baseline['base_head'])
        self.assertFalse((self.repo/'annealing/b1_prelude/validation_result.json').exists())

    def test_changed_candidate_tree_is_not_reconciled(self):
        original=publish.save
        def crash(path,value):
            if Path(path).name=='publication.json':raise OSError('after commit')
            return original(path,value)
        with patch.object(publish,'save',side_effect=crash):
            with self.assertRaises(OSError):publish.cycle(self.root,execute=True)
        git_at(self.repo,'rm','--cached','--','annealing/b1_prelude/a.py')
        git_at(self.repo,'commit','--amend','--no-edit')
        with self.assertRaises(ValueError):publish.cycle(self.root,execute=True)
        self.assertEqual(self.remote_head(),self.baseline['base_head'])

if __name__=='__main__':unittest.main()
