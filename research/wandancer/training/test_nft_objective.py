import unittest
import torch
from nft_objective import nft_loss

class NFTTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(7)
        values=[torch.randn(2,3,4,requires_grad=True) for _ in range(5)]
        return (*values,torch.tensor([.25,.75]),torch.tensor([-2.,3.]))

    def test_official_formula_and_detachment(self):
        xt,x0,current,old,reference,sigma,advantage=self.fixture()
        loss,metrics=nft_loss(xt,x0,sigma,current,old,reference,advantage)
        dims=(1,2); t=sigma.view(-1,1,1); target=x0.detach()
        positive=.1*current+.9*old.detach()
        negative=1.1*old.detach()-.1*current
        xp=xt.detach()-t*positive; xn=xt.detach()-t*negative
        with torch.no_grad():
            wp=(xp.double()-target.double()).abs().mean(dims,keepdim=True).clamp(min=1e-5)
            wn=(xn.double()-target.double()).abs().mean(dims,keepdim=True).clamp(min=1e-5)
            r=(advantage.clamp(-5,5)/5/2+.5).clamp(0,1)
        lp=((xp-target)**2/wp).mean(dims); ln=((xn-target)**2/wn).mean(dims)
        expected=((r*lp+(1-r)*ln)/.1*5).mean()+.004*((current-reference.detach())**2).mean()
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertIsNotNone(current.grad)
        self.assertTrue(torch.isfinite(current.grad).all())
        for x in [xt,x0,old,reference]:self.assertIsNone(x.grad)
        self.assertFalse(metrics['policy'].requires_grad)

    def test_zero_error_denominator_is_finite(self):
        x=torch.zeros(2,3,4); current=x.clone().requires_grad_()
        loss,_=nft_loss(x,x,torch.tensor([.5,.75]),current,x,x,torch.tensor([-5.,5.]))
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(current.grad).all())

    def test_advantage_clipping(self):
        xt,x0,current,old,reference,sigma,advantage=self.fixture()
        a,_=nft_loss(xt,x0,sigma,current,old,reference,torch.tensor([-100.,100.]))
        b,_=nft_loss(xt,x0,sigma,current,old,reference,torch.tensor([-5.,5.]))
        torch.testing.assert_close(a,b)

    def test_bad_shapes_and_sigma_rejected(self):
        xt,x0,current,old,reference,sigma,advantage=self.fixture()
        with self.assertRaises(ValueError):nft_loss(xt,x0,sigma,current[:1],old,reference,advantage)
        with self.assertRaises(ValueError):nft_loss(xt,x0,torch.tensor([-1.,2.]),current,old,reference,advantage)
        with self.assertRaises(ValueError):nft_loss(xt,x0,sigma,current,old,reference,advantage,beta=0)

if __name__=='__main__':unittest.main()
