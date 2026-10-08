"""Small CPU engineering checks; no model weights or clinical labels."""
import ast
import copy
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from core import INPUT_KEYS, objective, extract_answer
from p3_readout import option_log_probabilities, option_tokens
from p8_diagnostics import components, diagnostic_state, exact_kl, gradient_metrics


class Checks(unittest.TestCase):
    def test_exact_kl(self):
        p=torch.tensor([.1,.2,.7],dtype=torch.float64)
        q=torch.tensor([.3,.4,.3],dtype=torch.float64)
        self.assertAlmostEqual(float(exact_kl(p.log(),q.log())),float((p*(p/q).log()).sum()),12)
        self.assertEqual(float(exact_kl(q.log(),q.log())),0.)

    def test_components_and_scaling(self):
        theta=torch.tensor([.2,-.3,.1],requires_grad=True)
        entropy=theta.square()+.5
        kl=theta.square()
        mask=torch.tensor([True,False,True])
        cfg={"clip_epsilon":.2,"beta_lower":.01,"beta_upper":.01,"kl_coefficient":1.,"method":"SPINE"}
        args=(theta,theta.detach(),entropy,kl,torch.tensor(.4),mask,torch.tensor(.1),torch.tensor(.4),3,2)
        parts=components(*args,.2)
        loss,_=objective(*args,cfg)
        weighted=parts[0]+.01*parts[1]+parts[2]
        torch.testing.assert_close(weighted,loss)
        individual=[torch.autograd.grad(v,theta,retain_graph=True)[0] for v in parts]
        torch.testing.assert_close(sum([individual[0],.01*individual[1],individual[2]]),torch.autograd.grad(loss,theta,retain_graph=True)[0])
        torch.testing.assert_close(torch.autograd.grad(2*parts[2],theta,retain_graph=True)[0],2*individual[2])
        zeros=components(*args[:4],torch.tensor(0.),*args[5:],.2)
        self.assertTrue(torch.equal(torch.autograd.grad(zeros[0],theta)[0],torch.zeros_like(theta)))
        metrics=gradient_metrics([torch.zeros(3)]*3,.01,1.)
        self.assertIsNone(metrics['policy_band_cosine']);self.assertIsNone(metrics['band_policy_ratio'])

    def test_complete_option_sequences(self):
        class Model(torch.nn.Module):
            def forward(self,input_ids,attention_mask,**kwargs):
                logits=torch.arange(8,dtype=torch.float32).repeat(1,input_ids.shape[-1],1)
                return SimpleNamespace(logits=logits)
        encoded={'input_ids':torch.tensor([[1,2]]),'attention_mask':torch.ones(1,2,dtype=torch.long)}
        candidates={'A':[3],'B':[4,5]}
        result=option_log_probabilities(Model(),encoded,candidates)
        logp=torch.arange(8,dtype=torch.float32).log_softmax(-1)
        self.assertAlmostEqual(result['A'],float(logp[3]),6)
        self.assertAlmostEqual(result['B'],float(logp[4]+logp[5]),6)
        class Tokenizer:
            def encode(self,text,**kwargs):return [ord(c) for c in text]
        self.assertEqual(option_tokens(Tokenizer(),'prefix',{'AA':'x'})['AA'],[32,65,65])

    def test_state_and_input_boundary(self):
        model=torch.nn.Linear(2,2)
        model.register_buffer('constant',torch.ones(2))
        model.train();before=copy.deepcopy(model.state_dict()); rng=torch.get_rng_state().clone()
        with diagnostic_state(model,gradient=True):
            random.random();np.random.rand();torch.rand(2)
            torch.autograd.grad(model(torch.ones(1,2)).sum(),list(model.parameters()))
        self.assertTrue(model.training);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,model.state_dict()[n]) for n,v in before.items()))
        source=Path(__file__).with_name('p8_diagnostics.py').read_text()
        self.assertNotIn('read_labels',source)
        self.assertNotIn('evaluation_labels',source)
        tree=ast.parse(source)
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step' for n in ast.walk(tree)))
        self.assertNotIn('label',INPUT_KEYS)
        from p8_diagnostics import encode
        with self.assertRaises(ValueError):encode(None,{**dict.fromkeys(INPUT_KEYS),'label':'A'},True)
        self.assertIsNone(extract_answer('clinical prose only',{'A':'x','B':'y'}))


if __name__=='__main__':unittest.main()
