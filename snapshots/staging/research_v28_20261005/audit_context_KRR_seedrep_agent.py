"""Independent original-NB replay for 12 new selected-KRR matrices at two later fixed seeds.

Never imports producer or clean generator. No fit, truth read, new countfile,
or judge. Parent supplies exact complete protocol/seal hashes for one run.
"""
from pathlib import Path
import argparse, ast, ctypes, hashlib, json, os, shutil, sys, time, traceback
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='4'
import numpy as np
import anndata as ad
import h5py
from anndata.io import read_elem
import copy
import psutil
import scipy
from scipy import sparse
from threadpoolctl import threadpool_info,threadpool_limits

R=Path('C:/00 - Competition');O=R/'runs/research_v28_20261005/context_KRR_seedrep_conditional_v1'
M=R/'runs/research_v28_20261005/ntc_context_krr_native_mean_v1'
EXE=R/'envs/foundation312/Scripts/python.exe'
BASE='clean_B0_NB';NEW='context_KRR_NB';ARMS=(BASE,NEW)
SEEDS=(20260928,20260929,20260930);MAX_SECONDS=900.
PILOT=R/'runs/research_v28_20261005/ntc_context_krr_native400_v1'
NBROOT=R/'runs/research_v28_20261005/clean400_seedrep_v1'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*2**20):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def arr(p,keys=None):
    with np.load(p,allow_pickle=False) as z:return {k:np.asarray(z[k]).copy() for k in (z.files if keys is None else keys)}

def ah(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()

def ch(a):
    a=a.tocsr(copy=True);a.sort_indices();a.eliminate_zeros();h=hashlib.sha256();h.update(np.asarray(a.shape,np.int64).tobytes())
    for x in (a.indptr.astype(np.int64),a.indices.astype(np.int32),a.data.astype(np.float32)):h.update(x.tobytes())
    return h.hexdigest()

def rms(a):return float(np.sqrt(np.mean(np.asarray(a,dtype=float)**2)))

def check(p,h):assert sha(p)==h,str(p)

def fresh(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def guard(start):
    mem=psutil.Process().memory_info();peak=getattr(mem,'peak_wset',mem.rss)
    assert peak<2*2**30 and time.monotonic()-start<MAX_SECONDS and shutil.disk_usage(R).free>=50*2**30

def original_AST():
    folder=R/'archive/legacy_20261003/src/vcc2026';out={}
    names={'residual_v6_common.py':{'positions','source_gene_set','count_ratio'},'generate_counts.py':{'transform_counts','multiplier_from_effects'},'latent_counts_v26.py':{'integer_sample'},'prepare_response_inputs_v23.py':{'own_for'}}
    for name,wanted in names.items():
        path=folder/name;tree=ast.parse(path.read_text(encoding='utf8'));nodes={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted};assert nodes.keys()==wanted
        out[str(path)]=dict(file_sha256=sha(path),function_AST_SHA256={k:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for k,n in nodes.items()})
    path=folder/'literal_nb_v28.py';cls=next(n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n,ast.ClassDef) and n.name=='PanelGenerator')
    node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='build_target')
    out[str(path)]=dict(file_sha256=sha(path),build_target_AST_SHA256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest())
    return out

def dependencies(p):
    for path,h in p['dependencies'].items():check(path,h)
    for c in p['cases']:
        for s in c['base_case']['leaves'].values():check(s['path'],s['sha256'])
        check(c['mean_bundle']['bundle_path'],c['mean_bundle']['bundle_sha256'])
    for r in p['reuse_records']:
        for s in r['files'].values():check(s['path'],s['sha256'])

def runtime_check(p):
    """Producer-used DLLs are strict subset; extra audit BLAS bound separately."""
    from ctypes import wintypes
    f=ctypes.windll.kernel32.GetModuleFileNameW;f.argtypes=[wintypes.HMODULE,wintypes.LPWSTR,wintypes.DWORD];f.restype=wintypes.DWORD
    buf=ctypes.create_unicode_buffer(32768);assert f(ctypes.pythonapi._handle,buf,len(buf))>0
    actual=str(Path(buf.value).resolve());records=p['runtime']['loaded_binary_receipt']['actual_binaries']
    spec=records['actual_loaded_python312_DLL'];assert Path(actual)==Path(spec['path']).resolve();check(actual,spec['sha256'])
    actual_blas={str(Path(v['filepath']).resolve()):sha(v['filepath']) for v in threadpool_info() if v['user_api']=='blas'}
    expected={str(Path(v['path']).resolve()):v['sha256'] for k,v in records.items() if k.startswith('actual_loaded_BLAS::')}
    assert expected and expected.keys()<=actual_blas.keys()
    for path,h in expected.items():assert actual_blas[path]==h and p['generation_runtime_assets'][path]==h
    # scipy/AnnData may load additional DLLs; these are audit dependencies, not producer dependencies.
    extras={q:h for q,h in actual_blas.items() if q not in expected}
    for q,h in extras.items():assert 'scipy.libs' in q.replace(chr(92),'/');check(q,h)
    return dict(actual_python312_DLL=actual,producer_BLAS_SHA256=expected,audit_extra_BLAS_SHA256=extras,python=sys.version,numpy=np.__version__,scipy=scipy.__version__)

def ratio(effect,mean,model,native,source_genes):
    lookup={g:i for i,g in enumerate(native)}
    src=np.asarray([i for i,g in enumerate(model) if g in lookup and (g in source_genes or effect[i]!=0)],np.int64)
    dst=np.asarray([lookup[model[i]] for i in src],np.int64)
    mu=mean[dst];wanted=np.maximum((mu+.5)*np.exp2(effect[src])-.5,0.)
    values=np.ones(len(dst));positive=mu>0;values[positive]=wanted[positive]/mu[positive]
    out=np.ones(len(native));out[dst]=np.clip(values,0,16);return out

def H0(donor,rho,ordinal,seed):
    out=donor.tocsr(copy=True);value=out.data.astype(float)*rho[out.indices];floor=np.floor(value)
    rng=np.random.default_rng(np.random.SeedSequence([seed,ordinal,0]))
    out.data=(floor+(rng.random(len(value))<value-floor)).astype(np.float32);out.eliminate_zeros();out.sort_indices();return out

def NB(mu,theta,ordinal,seed):
    rng=np.random.default_rng(np.random.SeedSequence([seed,ordinal,26,3]));before=rng.bit_generator.state
    value=rng.negative_binomial(theta[None,:],theta[None,:]/(theta[None,:]+mu)).astype(np.float32)
    return value,dict(sampler='nb',seed_sequence_entropy=[seed,ordinal,26,3],rng_before=before,rng_after=rng.bit_generator.state,paired_gene_uniforms_claimed=False,NB_rejection_sampler_gene_pairing_claimed=False)

def normalized(a):
    L=a.sum(1);assert np.isfinite(L).all() and (L>0).all();cp=a/L[:,None]*1e6
    return cp.mean(0),np.log1p(cp).mean(0),L

def metadata_axis(path,native,targets):
    """Read obs/var only, never firstseed X. All counts opaque-SHA locked."""
    with h5py.File(path,'r') as h:
        obs=read_elem(h['obs']);var=read_elem(h['var']);shape=tuple(h['X'].attrs['shape'])
    assert shape==(3200,len(native)) and list(obs.columns)==['target_gene']
    np.testing.assert_array_equal(var.index.to_numpy().astype(str),native)
    assert obs.target_gene.astype(str).tolist()==[t for t in targets for _ in range(400)]
    assert len(obs.index)==len(set(obs.index))==3200

def verify_manifest(r,ph,extra):
    assert r['completed'] and r['protocol_sha256']==ph and r['seed'] in SEEDS
    assert r['count_seed']==r['seed'] and r['donor_seed']==r['seed']+100
    assert r['cells']==3200 and r['cells_per_target']==400
    for name in ('prediction','moments','generation_manifest'):
        check(r[name+'_path'],r[name+'_sha256']);extra[r[name+'_path']]=r[name+'_sha256']
    m=read(r['generation_manifest_path']);rr=dict(r);rr.pop('generation_manifest_path');rr.pop('generation_manifest_sha256');assert m==rr

def independent_block(donor,effect,mean,model,native,source_genes,eo,nat,th,ordinal,seed):
    D=donor.toarray().astype(float);rho=ratio(effect,mean,model,native,source_genes)
    literal=D*rho[None,:];B=literal[:,nat].copy();h0=H0(donor,rho,ordinal,seed)
    draw,info=NB(B,th,ordinal,seed);expected=h0.toarray();expected[:,nat]=draw
    protected=np.ones(len(native),bool);protected[nat]=False
    conditional=literal.copy();conditional[:,protected]=h0[:,protected].toarray().astype(float)
    return D,rho,literal,B,h0,expected,conditional,protected,info

def check_target_hashes(d,block,rho,literal,B,h0,conditional,donor_rows,eo,info):
    assert d['literal_conditional_mean_equal_H0_B'] and d['protected_integer_equality_H0']
    assert ah(rho)==d['count_ratio_native_sha256'] and ah(B)==d['literal_B_editable_sha256']==d['conditional_mean_editable_sha256']
    assert ah(literal)==d['literal_B_native_sha256'] and ah(conditional)==d['conditional_given_H0_native_mean_sha256']
    assert ah(donor_rows)==d['donor_pool_rows_sha256'] and ah(eo)==d['editable_output_positions_sha256']
    assert ch(block)==d['count_block_sha256'] and ch(h0)==d['H0_count_block_sha256'] and info==d['sampler']

def run(ph,sh):
    assert Path(sys.executable).resolve()==EXE.resolve() and np.__version__=='2.5.3';threadpool_limits(4)
    check(O/'PROTOCOL.json',ph);check(O/'PREDICTIONS_SEALED.json',sh)
    assert not (O/'GENERATING.lock').exists() and not (O/'SCORING.lock').exists()
    assert not (O/'SCORE_PROTOCOL.json').exists(),'This gate runs before any new judge'
    p=read(O/'PROTOCOL.json');s=read(O/'PREDICTIONS_SEALED.json');gc=read(O/'GENERATION_COMPLETE.json')
    assert p['schema']=='conditional_selected_context_KRR_originalNB400_seedrep_v1' and p['frozen'] and not p['pending_bindings']
    assert p['seeds']==list(SEEDS) and p['selected_arm']==NEW and p['arms']==list(ARMS) and len(p['cases'])==6 and p['cells_per_target']==400
    assert p['newfits']==p['newmeanforwards']==p['THP_reads']==p['official_calls']==0 and not p['truth_read_by_generation']
    assert (p['total_outputs'],p['new_outputs'],p['reused_outputs'],p['reused_B0_outputs'],p['reused_selected_pilot_outputs'])==(36,12,24,18,6)
    assert p['budget']==dict(CPU_threads=4,numeric_seconds=600.0,RSS_GiB=2,new_prediction_bytes=512*2**20,C_free_GiB=50)
    assert s['completed'] and s['protocol_sha256']==ph and s['all36_before_new_judge'] and s['newjudge_at_seal']==0
    assert (s['total_outputs'],s['new_outputs'],s['reused_outputs'],s['selected_arm'])==(36,12,24,NEW)
    assert gc['completed'] and gc['prediction_seal_sha256']==sh and gc['new_outputs']==12 and gc['reused_outputs']==24 and gc['newjudge']==0
    assert gc['resources']['new_prediction_bytes']<=512*2**20
    mg=read(M/'MEAN_INDEPENDENT_AUDIT.json');fg=read(PILOT/'GENERATION_INDEPENDENT_AUDIT.json');nbp=read(NBROOT/'PROTOCOL.json')
    assert mg['completed'] and mg['passed'] and mg['mean_seal_sha256']==sha(M/'MEANS_SEALED.json')==p['mean_seal_sha256']
    assert sha(M/'MEAN_INDEPENDENT_AUDIT.json')==p['mean_gate_sha256'] and sha(M/'PROTOCOL.json')==p['mean_protocol_sha256']
    assert fg['completed'] and fg['passed'] and fg['target_blocks_full_replayed']==144 and fg['original_B0_NB_H0_blocks_independently_replayed']==48
    assert fg['prediction_seal_sha256']==sha(PILOT/'PREDICTIONS_SEALED.json') and fg['protocol_sha256']==sha(PILOT/'PROTOCOL.json')
    ticket=read(p['choice_ticket_path']);check(p['choice_ticket_path'],p['choice_ticket_sha256'])
    pilotp=read(PILOT/'PROTOCOL.json');post=read(PILOT/'POSTSCORE_INDEPENDENT_AUDIT.json')
    assert post['completed'] and post['passed'] and len(post['all24'])==24
    assert sha(PILOT/'COMPLETE.json')==p['firstscreen_COMPLETE_sha256']==post['complete_sha256']
    assert sha(PILOT/'POSTSCORE_INDEPENDENT_AUDIT.json')==p['firstscreen_postscore_audit_sha256']
    assert p['evaluation_sha256']==ticket['evaluation_sha256']==pilotp['evaluation_sha256']=='c8a719bda010e352af4dd7c433313c795eab5460e709897a95b474cd9091e35c'
    assert p['metric_weights']==pilotp['metric_weights']==nbp['source_clean400_protocol']['metric_weights']
    assert p['original_evaluation_aggregation']==pilotp['aggregation'] and len(p['metric_weights'])==6 and all(w==1/6 for w in p['metric_weights'].values())
    pre=time.monotonic();dependencies(p);pre_seconds=time.monotonic()-pre
    start=time.monotonic();guard(start);ast_contract=original_AST();assert ast_contract==fg['original_scientific_AST_start_end'];runtime=runtime_check(p)
    by={(r['seed'],r['role'],r['arm']):r for r in s['records']};assert len(by)==36
    assert set(by)=={(seed,c['role'],arm) for seed in SEEDS for c in p['cases'] for arm in ARMS}
    oldby={(r['seed'],r['role'],r['arm']):r for r in p['reuse_records']};assert len(oldby)==24
    assert set(oldby)=={(seed,c['role'],BASE) for seed in SEEDS for c in p['cases']}|{(SEEDS[0],c['role'],NEW) for c in p['cases']}
    source_path=R/'data/processed/v1/external_pseudobulk/replogle_k562_gwps.npz'
    with np.load(source_path,allow_pickle=False) as zsrc:source_genes=set(zsrc['gene_names'].astype(str))
    extra={str(O/'PROTOCOL.json'):ph,str(O/'PREDICTIONS_SEALED.json'):sh,str(O/'GENERATION_COMPLETE.json'):sha(O/'GENERATION_COMPLETE.json')}
    for r in s['records']:verify_manifest(r,ph,extra)
    reuse_files=score_files=old_assets=0;rows=[];baseline_rows=[];reused_identity_records=[]
    for key,old in oldby.items():
        seed,role,arm=key;r=by[key];dest=O/f'seed_{seed}'/role
        assert r['strict_reuse'] and r['source_identity']==old['source_record']['source_identity'] and r['source_files']==old['files']
        assert r['source_protocol_sha256']==old['source_protocol_sha256'] and r['scores_reused_no_new_judge'] and r['newcount']==r['newjudge']==0
        for name,b in old['files'].items():
            check(b['path'],b['sha256']);extra[b['path']]=b['sha256'];reuse_files+=1;score_files+=name.startswith('score/')
            path=dest/'scores'/arm/name.removeprefix('score/') if name.startswith('score/') else dest/({'prediction':f'pred__{arm}.h5ad','moments':f'{arm}_moments.npz','generation':f'{arm}_SOURCE_generation.json'}[name])
            check(path,b['sha256']);extra[str(path)]=b['sha256']
        cw=next(c for c in p['cases'] if c['role']==role);native=np.loadtxt(cw['base_case']['leaves']['genes.txt']['path'],dtype=str,ndmin=1)
        metadata_axis(r['prediction_path'],native,cw['base_case']['targets'])
        reused_identity_records.append(dict(seed=seed,role=role,arm=arm,all_source_local_files_SHA_exact=True,original_obs_var_metadata_exact=True,firstseed_X_not_decoded=seed==SEEDS[0]))
    assert reuse_files==294 and score_files==222,'Actual protocol includes 294 old assets; source/local wrapper manifests are counted separately'
    for seed in SEEDS[1:]:
        for cw in p['cases']:
            guard(start);c=cw['base_case'];role=c['role'];dest=O/f'seed_{seed}'/role;br=by[seed,role,BASE];r=by[seed,role,NEW];old=oldby[seed,role,BASE]
            assert not r['strict_reuse'] and r['truth_read_by_generation']==False and r['newfits']==r['newjudge']==r['official_calls']==0
            for name,b in c['leaves'].items():
                if name in ('real.h5ad','panel.json','genes.txt','generation_pool.npz') or name.startswith('reference'):
                    rel='references/'+name[len('reference/'):] if name.startswith('reference/') else name
                    check(dest/rel,b['sha256']);extra[str(dest/rel)]=b['sha256'];old_assets+=1
            z=arr(c['leaves']['B0_effects']['path']);theta=arr(c['leaves']['theta_seen']['path']);input_genes=arr(c['leaves']['role_encoder']['path'],['genes'])['genes'].astype(str)
            b=cw['mean_bundle'];check(b['bundle_path'],b['bundle_sha256']);value=arr(b['bundle_path']);assert value.keys()==z.keys() and b['arm']=='context_KRR'
            for name in z:
                if name!='effect':np.testing.assert_array_equal(value[name],z[name]);assert value[name].dtype==z[name].dtype
            assert value['effect'].dtype==np.float32 and np.isfinite(value['effect']).all() and not value['effect'][~z['editable_mask'].astype(bool)].any()
            native=np.loadtxt(c['leaves']['genes.txt']['path'],dtype=str,ndmin=1);model=z['genes'].astype(str);targets=list(c['targets'])
            np.testing.assert_array_equal(native,z['native_genes'].astype(str));np.testing.assert_array_equal(theta['gene_symbols'].astype(str),model)
            np.testing.assert_array_equal(theta['seen'].astype(bool),z['new_generator_seen'].astype(bool));assert np.isfinite(theta['theta']).all() and (theta['theta']>0).all()
            np.testing.assert_array_equal(z['ownmask'],(np.asarray(targets)[:,None]!=model[None,:]).astype(np.float32))
            assert str(z['effect_units'])=='raw_log2_pseudocount_0.5' and not bool(z['held_response_used']) and not bool(z['historical_fallback_used'])
            assert set(z['source_contexts'].astype(str))<=set(c['allowed_response_contexts'])
            pool=sparse.load_npz(c['leaves']['generation_pool.npz']['path']).tocsr();assert pool.shape==(3000,len(native)) and pool.dtype==np.float32
            assert np.isfinite(pool.data).all() and (pool.data>=0).all() and np.equal(pool.data,np.rint(pool.data)).all()
            mean=np.asarray(pool.mean(0)).ravel().astype(float);nmap={g:i for i,g in enumerate(native)};mmap={g:i for i,g in enumerate(model)}
            op=np.asarray([mmap[g] for g in input_genes],np.int64);assert len(op)==len(set(input_genes))==6119
            order=list(map(str,read(c['leaves']['panel.json']['path']).get('generation_targets',targets)));assert order==cw['original_generation_target_order'] and len(order)==8 and set(order)==set(targets)
            dr=np.random.default_rng(seed+100);donors={t:dr.choice(3000,400,replace=False) for t in order}
            check(r['donor_receipt_path'],r['donor_receipt_sha256']);extra[r['donor_receipt_path']]=r['donor_receipt_sha256']
            receipt=read(r['donor_receipt_path']);assert receipt==dict(seed=seed,donor_seed=seed+100,pool_rows=3000,pool_SHA256=c['leaves']['generation_pool.npz']['sha256'],original_order=order,draws={t:donors[t].tolist() for t in order})
            identity=copy.deepcopy(br['source_identity']);identity.update(mean_path=b['bundle_path'],mean_sha256=b['bundle_sha256'],mean_workflow=b['arm'],context_mean_protocol_path=str(M/'PROTOCOL.json'),context_mean_protocol_SHA256=sha(M/'PROTOCOL.json'),source_model_seal_path=b['source_model_seal_path'],source_model_seal_SHA256=b['source_model_seal_SHA256'],baseline_same_seed_prediction_path=br['prediction_path'],baseline_same_seed_prediction_sha256=br['prediction_sha256'],baseline_same_seed_generation_path=br['generation_manifest_path'],baseline_same_seed_generation_sha256=br['generation_manifest_sha256'],same_seed_original_donor_receipt_path=r['donor_receipt_path'],same_seed_original_donor_receipt_sha256=r['donor_receipt_sha256'])
            assert identity==r['source_identity'] and not identity['destination_truth_loaded'] and not identity['legacy_effect_theta_encoder_numeric_loaded']
            baseline=ad.read_h5ad(br['prediction_path']);base_mm=arr(br['moments_path']);oldgen=read(old['files']['generation']['path']);odby={d['target']:d for d in oldgen['targets']}
            obj=ad.read_h5ad(r['prediction_path']);mm=arr(r['moments_path']);X=obj.X.tocsr()
            assert obj.shape==baseline.shape==(3200,len(native)) and obj.obs_names.tolist()==[f'{NEW}_{i}' for i in range(3200)] and list(obj.obs.columns)==['target_gene']
            assert obj.obs.target_gene.astype(str).tolist()==baseline.obs.target_gene.astype(str).tolist()==[t for t in targets for _ in range(400)]
            np.testing.assert_array_equal(obj.var_names.to_numpy().astype(str),native);np.testing.assert_array_equal(baseline.var_names.to_numpy().astype(str),native)
            for a in (mm,base_mm):np.testing.assert_array_equal(a['genes'].astype(str),native);np.testing.assert_array_equal(a['targets'].astype(str),targets)
            assert set(mm)=={'genes','targets','expected_and_sample_mean'} and mm['expected_and_sample_mean'].shape==(8,2,len(native))
            assert X.dtype==np.float32 and np.isfinite(X.data).all() and (X.data>0).all() and (X.data<=2**24).all() and np.equal(X.data,np.rint(X.data)).all()
            assert len(r['targets'])==len(r['original_donor_receipts'])==len(odby)==8
            for ti,t in enumerate(targets):
                guard(start);ordinal=order.index(t);eo=op[theta['seen'].astype(bool)[op]&z['ownmask'][ti].astype(bool)[op]];nat=np.asarray([nmap[model[j]] for j in eo],np.int64);th=theta['theta'].astype(float)[eo]
                donor=pool[donors[t]].tocsr(copy=True);baseblock=baseline.X[ti*400:(ti+1)*400].tocsr();base=baseblock.toarray().astype(float);od=odby[t]
                D,rho0,literal0,B0,h00,expected0,conditional0,protected,info0=independent_block(donor,z['effect'][ti],mean,model,native,source_genes,eo,nat,th,ordinal,seed)
                np.testing.assert_array_equal(base,expected0);check_target_hashes(od,baseblock,rho0,literal0,B0,h00,conditional0,donors[t],eo,info0)
                np.testing.assert_array_equal(base_mm['expected_and_sample_mean'][ti,0],conditional0.mean(0))
                np.testing.assert_array_equal(base_mm['expected_and_sample_mean'][ti,1],np.asarray(baseblock.mean(0)).ravel().astype(float))
                baseline_rows.append(dict(seed=seed,role=role,target=t,original_B0_H0_NB_exact=True,count_block_sha256=ch(baseblock),donor_sha256=ah(donors[t]),sampler_namespace=[seed,ordinal,26,3]))
                D,rho,literal,B,h0,expected,conditional,protected,info=independent_block(donor,value['effect'][ti],mean,model,native,source_genes,eo,nat,th,ordinal,seed)
                block=X[ti*400:(ti+1)*400];actual=block.toarray().astype(float);d=r['targets'][ti]
                assert d['target']==t and d['arm']==NEW and d['cells']==400 and d['same_original_donor_receipt_sha256']==r['donor_receipt_sha256']
                assert r['original_donor_receipts'][ti]==dict(target=t,donor_rows=donors[t].astype(int).tolist())
                np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(actual[:,protected],h0[:,protected].toarray());np.testing.assert_array_equal(actual[:,protected],base[:,protected])
                assert d['protected_counts_equal_same_seed_B0'] and d['zero_literal_mu_emits_zero'] and (actual[:,nat][B==0]==0).all() and d['editable_new_nonzero_coordinates']==0
                check_target_hashes(d,block,rho,literal,B,h0,conditional,donors[t],eo,info)
                np.testing.assert_array_equal(mm['expected_and_sample_mean'][ti,0],conditional.mean(0));sparse_saved=np.asarray(block.mean(0)).ravel().astype(float)
                np.testing.assert_array_equal(mm['expected_and_sample_mean'][ti,1],sparse_saved)
                library_delta=float(conditional.mean(0).sum()-base_mm['expected_and_sample_mean'][ti,0].sum());assert library_delta==d['conditional_expected_library_delta_vs_same_seed_B0']
                cpD,logD,libD=normalized(D);cpB,logB,libB=normalized(base);cpY,logY,libY=normalized(actual);cpE,logE,libE=normalized(conditional)
                assert (libY<1e6).all();np.testing.assert_array_equal(np.asarray(block.sum(1)).ravel(),libY)
                rows.append(dict(seed=seed,role=role,context=c['context'],target=t,arm=NEW,independent_original_H0_NB_integer_exact=True,source_mean_theta_seen_own_donor_order_exact=True,protected_H0_and_same_seed_B0_exact=True,zero_carrier_support_exact=True,
                    conditional_expected_mean_F64_exact=True,saved_sparse_F32_sample_mean_exact=True,saved_sample_precision_difference_maxabs=float(np.abs(actual.mean(0)-sparse_saved).max(initial=0)),full_native_axis_profile_recomputed_by_audit=True,
                    count_block_sha256=ch(block),donor_pool_rows_sha256=ah(donors[t]),editable_genes=len(nat),protected_genes=int(protected.sum()),theta_sha256=ah(th),sampler_namespace=[seed,ordinal,26,3],
                    actual_raw_mean_RMS_vs_conditional=rms(actual.mean(0)-conditional.mean(0)),actual_CP1Mmean_RMS_vs_B0=rms(cpY-cpB),actual_meanlogCP1M_RMS_vs_B0=rms(logY-logB),conditional_CP1Mmean_RMS_vs_B0=rms(cpE-cpB),actual_library_RMS_vs_B0=rms(libY-libB),actual_mean_library=float(libY.mean()),conditional_expected_library_delta_vs_B0=library_delta,normalized_invariance_claimed=False))
            print('KRR_SEEDREP_INDEPENDENT',seed,role,'newblocks',len(rows),'same_seed_B0blocks',len(baseline_rows),flush=True)
            del obj,X,baseline,pool
    assert len(rows)==len(baseline_rows)==96;guard(start)
    numeric_elapsed=time.monotonic()-start;post_start=time.monotonic();dependencies(p)
    for path,h in extra.items():check(path,h)
    assert original_AST()==ast_contract and runtime_check(p)==runtime
    assert not (O/'SCORE_PROTOCOL.json').exists() and not (O/'GENERATING.lock').exists()
    report=dict(completed=True,passed=True,protocol_sha256=ph,prediction_seal_sha256=sh,target_blocks_full_replayed=96,candidate_matrices_full_replayed=12,original_B0_NB_H0_blocks_independently_replayed=96,strict_reused_matrices=24,strict_reused_B0_matrices=18,strict_reused_firstseed_selected_matrices=6,
        firstseed_count_values_replayed_again=False,firstseed_closed_generation_gate_sha256=sha(PILOT/'GENERATION_INDEPENDENT_AUDIT.json'),firstseed_closed_postscore_gate_sha256=sha(PILOT/'POSTSCORE_INDEPENDENT_AUDIT.json'),baseline_reuse_file_SHA_checks=reuse_files,baseline_score_file_SHA_checks=score_files,old_reference_panel_axis_NTC_assets_SHA_checks=old_assets,dependencies_unchanged_start_end=len(p['dependencies']),source_local_scores_outputs_unchanged_start_end=len(extra),reuse_metadata_records=reused_identity_records,
        evaluation_sha256=p['evaluation_sha256'],metric_weights=p['metric_weights'],original_evaluation_aggregation=p['original_evaluation_aggregation'],choice_ticket_sha256=p['choice_ticket_sha256'],mean_gate_sha256=p['mean_gate_sha256'],mean_seal_sha256=p['mean_seal_sha256'],audit_code_path=str(Path(__file__).resolve()),audit_code_sha256=sha(__file__),runtime=runtime,original_scientific_AST_start_end=ast_contract,records=rows,original_B0_records=baseline_rows,
        producer_or_clean_generator_imported=False,original_count_formulas='Independent rawPC.5 inverse/floor/cap; original sparseF32 pool mean promotedF64; sparse H0 stochasticround; literal Drho NB2 theta/(theta+mu); seed+100 complete original-order donors; seed,ordinal,26,3 NB namespace.',
        precision='Conditional_given_H0 F64 mean exact; sparseF32 sample mean promotedF64 exact. Actual denseintegerF64 bulk, fullaxis CP1M arithmetic/meanlog profiles and celllibraries independently checked; no normalized invariance.',
        scope='Only two later fixed countseeds at all6 originalroles/all48 developed Native targets; technical countrepetitions. No sourceT or destination perturbation count values. Firstseed counts/scores strictSHA reuse and closed144/48 gate only.',
        elapsed_seconds=numeric_elapsed,opaque_preverify_seconds=pre_seconds,opaque_postverify_seconds=time.monotonic()-post_start,peak_RSS_bytes=getattr(psutil.Process().memory_info(),'peak_wset',psutil.Process().memory_info().rss),newcountfiles=0,newfits=0,newjudges=0,Native_held_perturbation_values_read=0,THP_reads=0,official_calls=0,replay_draws_in_memory_only=True)
    path=O/'GENERATION_INDEPENDENT_AUDIT.json';fresh(path,report);print(json.dumps(dict(passed=True,path=str(path),sha256=sha(path))),flush=True)

if __name__=='__main__':
    cli=argparse.ArgumentParser();cli.add_argument('--protocol-sha',required=True);cli.add_argument('--seal-sha',required=True);a=cli.parse_args()
    try:run(a.protocol_sha,a.seal_sha)
    except BaseException as e:
        fresh(O/f'GENERATION_INDEPENDENT_AUDIT_FAILED_AGENT_{time.time_ns()}.json',dict(completed=False,passed=False,error=repr(e),traceback=traceback.format_exc(),audit_code_sha256=sha(__file__),protocol_sha256=a.protocol_sha,prediction_seal_sha256=a.seal_sha,newcountfiles=0,newfits=0,newjudges=0,Native_truth_reads=0,THP_reads=0));raise

