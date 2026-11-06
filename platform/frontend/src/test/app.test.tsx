import {describe,it,expect,vi,beforeEach} from 'vitest';
import {render,screen,within,waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {QueryClient,QueryClientProvider} from '@tanstack/react-query';
import App from '../App';
import {api} from '../services/api';
vi.mock('../services/api',async()=>{const original=await vi.importActual<typeof import('../services/api')>('../services/api');return {...original,api:{...original.api,datasets:vi.fn(),analyses:vi.fn(),dataset:vi.fn(),upload:vi.fn(),sample:vi.fn(),suggest:vi.fn(),quickstart:vi.fn(),configure:vi.fn(),analyze:vi.fn(),train:vi.fn(),job:vi.fn(),analysis:vi.fn()}}});
const mocked=vi.mocked(api);
function show(){return render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false,gcTime:0}}})}><App/></QueryClientProvider>)}
const nav=()=>within(screen.getByRole('navigation',{name:'Main navigation'}));
const dataset:any={id:'bank-test',filename:'bank-test.csv',file_size:100,fingerprint:'test-fixture-hash',configuration:null,preview:[{CustomerId:1,Fee:20,Exited:0}],profile:{rows:100,columns_count:3,missing_cells:0,duplicate_rows:0,warnings:[],target_candidates:['Exited'],preview:[],columns:[{name:'CustomerId',inferred_type:'numeric',identifier:true,missing:0,unique:100,sample_values:[1,2]},{name:'Fee',inferred_type:'numeric',missing:0,unique:40,sample_values:[20,30]},{name:'Exited',inferred_type:'binary',missing:0,unique:2,sample_values:[0,1],target_labels:['0','1']}]}};
const overview={rows:100,total_rows:100,churn_rate:.2,churned:20,retained:80,label_available:true,missing_cells:0,columns_count:3,findings:[],segments:[],numeric_summaries:[]};
beforeEach(()=>{vi.resetAllMocks();localStorage.clear();location.hash='';mocked.datasets.mockResolvedValue({datasets:[]});mocked.analyses.mockResolvedValue({analyses:[]});mocked.analysis.mockImplementation(async(_id,endpoint)=>endpoint==='models'?Promise.reject(new Error('No fitted model')):overview);mocked.job.mockResolvedValue({id:'job-1',analysis_id:'analysis-test',dataset_id:'bank-test',status:'running',progress:40,message:'Fitting models'})});
describe('frontend navigation and automatic setup (explicitly mocked test fixtures)',()=>{
 it('preserves a separate IBM entry point and shows honest initial empty states',async()=>{
  show();expect(await screen.findByRole('heading',{name:'Customer retention analysis'})).toBeInTheDocument();expect(screen.getByRole('link',{name:/IBM Telco case study/})).toHaveAttribute('href','http://127.0.0.1:8501');
  await userEvent.click(nav().getByRole('button',{name:'Customers'}));expect(await screen.findByRole('heading',{name:'Choose an analysis to continue'})).toBeInTheDocument();expect(screen.queryByText('80.5%')).not.toBeInTheDocument();
 });
 it('rejects unsupported files without calling upload',async()=>{
  show();await userEvent.click(nav().getByRole('button',{name:'Data setup'}));const input=screen.getByLabelText('Upload dataset');await userEvent.upload(input,new File(['bad'],'notes.txt',{type:'text/plain'}),{applyAccept:false});expect(await screen.findByText('Use a CSV file or an Excel .xlsx workbook.')).toBeInTheDocument();expect(mocked.upload).not.toHaveBeenCalled();
 });
 it('uploads a file and runs the automatic setup with no form input',async()=>{
  mocked.upload.mockResolvedValue(dataset);mocked.dataset.mockResolvedValue(dataset);mocked.suggest.mockRejectedValue(new Error('not needed'));
  mocked.quickstart.mockResolvedValue({analysis_id:'analysis-test',dataset_id:'bank-test',mode:'historical',status:'queued',job_id:'job-1',configuration:{} as any,notes:[],excluded:[]});
  show();await userEvent.upload(await screen.findByLabelText('Upload dataset'),new File(['CustomerId,Fee,Exited\n1,20,0'],'bank-test.csv',{type:'text/csv'}));
  await waitFor(()=>expect(mocked.quickstart).toHaveBeenCalledWith('bank-test'));expect(mocked.configure).not.toHaveBeenCalled();
  expect(await screen.findByRole('heading',{level:1,name:'Insights'})).toBeInTheDocument();
 });
 it('loads the bundled sample in one click',async()=>{
  mocked.sample.mockResolvedValue(dataset);mocked.dataset.mockResolvedValue(dataset);mocked.suggest.mockRejectedValue(new Error('not needed'));
  mocked.quickstart.mockResolvedValue({analysis_id:'analysis-test',dataset_id:'bank-test',mode:'historical',status:'queued',job_id:'job-1',configuration:{} as any,notes:[],excluded:[]});
  show();await userEvent.click(await screen.findByRole('button',{name:'Use IBM Telco sample'}));await waitFor(()=>expect(mocked.quickstart).toHaveBeenCalledWith('bank-test'));
 });
 it('still allows manual schema adjustment, excluding the ID and training afterwards',async()=>{
  mocked.datasets.mockResolvedValue({datasets:[dataset]});mocked.dataset.mockResolvedValue(dataset);mocked.suggest.mockRejectedValue(new Error('no suggestion'));mocked.configure.mockResolvedValue({dataset_id:dataset.id});mocked.train.mockResolvedValue({analysis_id:'analysis-test',job_id:'job-1',status:'queued'});
  show();await screen.findByRole('option',{name:'bank-test.csv'});await userEvent.selectOptions(screen.getByLabelText('Dataset'),'bank-test');await userEvent.click(nav().getByRole('button',{name:'Data setup'}));await userEvent.click(await screen.findByRole('button',{name:/Adjust manually/}));await screen.findByRole('heading',{name:'Define the question'});
  await userEvent.selectOptions(screen.getByLabelText(/Historical outcome column/),'Exited');await userEvent.selectOptions(screen.getByLabelText(/Positive churn label/),'1');await userEvent.selectOptions(screen.getByLabelText(/Customer identifier/),'CustomerId');await userEvent.click(screen.getByRole('button',{name:/^Review predictors$/}));await userEvent.click(screen.getByRole('button',{name:/Next: business meaning/}));await userEvent.click(screen.getByRole('button',{name:/Save and run analysis/}));
  await waitFor(()=>expect(mocked.configure).toHaveBeenCalled());const payload=mocked.configure.mock.calls[0][1];expect(payload.target_column).toBe('Exited');expect(payload.positive_label).toBe('1');expect(payload.features).toEqual([{column:'Fee',kind:'numeric'}]);expect(payload.customer_id).toBe('CustomerId');expect(payload.semantics).toEqual([]);await waitFor(()=>expect(mocked.train).toHaveBeenCalledWith('bank-test',true,'pr_auc',42));
 });
});
