import {afterEach,describe,it,expect,vi} from 'vitest';
import {request,ApiError,params,api} from '../services/api';
afterEach(()=>vi.unstubAllGlobals());
describe('versioned API client contract',()=>{
 it('encodes category lists and numeric filters without mixing schemas',()=>{
  const query=params({filters:{Plan:['Annual'],Fee:{min:10}},model:'Hybrid Ensemble'});const parsed=new URLSearchParams(query);expect(JSON.parse(parsed.get('filters')!)).toEqual({Plan:['Annual'],Fee:{min:10}});expect(parsed.get('model')).toBe('Hybrid Ensemble');
 });
 it('displays real backend validation errors rather than a success state',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify({detail:[{loc:['body','positive_label'],msg:'Unsupported label'}]}),{status:422})));await expect(request('/datasets/x/configuration')).rejects.toThrow('positive_label: Unsupported label');
 });
 it('retains HTTP status for missing model artifacts',async()=>{
  vi.stubGlobal('fetch',vi.fn().mockResolvedValue(new Response(JSON.stringify({detail:'Train this analysis first'}),{status:409})));try{await request('/analyses/a/models');throw new Error('Expected failure')}catch(error){expect(error).toBeInstanceOf(ApiError);expect((error as ApiError).status).toBe(409)}
 });
 it('sends the explicitly confirmed labels and seed in training requests',async()=>{
  const fetcher=vi.fn().mockResolvedValue(new Response(JSON.stringify({analysis_id:'a',job_id:'j',status:'queued'})));vi.stubGlobal('fetch',fetcher);await api.train('dataset-a',true,'pr_auc',17);expect(fetcher.mock.calls[0][0]).toBe('/api/v1/analyses/train');expect(JSON.parse(fetcher.mock.calls[0][1].body)).toEqual({dataset_id:'dataset-a',fast:true,selection_metric:'pr_auc',random_state:17});
 });
 it('keeps dataset/model version and filters in export URLs',()=>{
  const url=api.exportUrl('analysis-v2','customers',{model:'XGBoost',risk:'High',search:'A&B'});expect(url).toContain('/analyses/analysis-v2/export');const values=new URLSearchParams(url.split('?')[1]);expect(values.get('risk')).toBe('High');expect(values.get('search')).toBe('A&B');
 });
});
