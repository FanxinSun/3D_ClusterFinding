// probe2.C -- TPC-only follow-up (single pass over each tree)
//   root -l -b -q 'probe2.C("clusters_seeds_island_RUN_REF-0.root_ntuplizer.root")'          // full ntp_hit (~3 min)
//   root -l -b -q 'probe2.C("file.root", 5000000)'                                          // first 5M hits only
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TF1.h"
#include "TStyle.h"
#include "TPad.h"
#include <set>
#include <map>
#include <tuple>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <algorithm>

static int reg(int L) { return L < 7 ? -1 : L <= 22 ? 0 : L <= 38 ? 1 : L <= 54 ? 2 : -1; }
static int lastBin(TH1* h) { for (int b = h->GetNbinsX(); b >= 1; --b) if (h->GetBinContent(b) > 0) return b; return -1; }
static void fitline(TProfile* p, const char* tag) {
  int lo = -1, hi = -1; for (int b = 1; b <= p->GetNbinsX(); ++b) if (p->GetBinEntries(b) > 20) { if (lo < 0) lo = b; hi = b; }
  if (lo < 0) { printf("  %-28s : empty\n", tag); return; }
  p->Fit("pol1", "Q0", "", p->GetBinCenter(lo) - 0.5, p->GetBinCenter(hi) + 0.5);
  TF1* fn = p->GetFunction("pol1");
  printf("  %-28s : %10.4f + %9.5f * tbin    (tbin %d..%d populated)   z(tbin=0)=%.1f  z(tbin=%d)=%.1f\n", tag, fn->GetParameter(0),
         fn->GetParameter(1), (int)p->GetBinCenter(lo), (int)p->GetBinCenter(hi), fn->Eval(0), (int)p->GetBinCenter(hi), fn->Eval(p->GetBinCenter(hi)));
}
struct BR { const char* n; Float_t* p; };
static void attach(TTree* t, std::vector<BR>& v) { t->SetBranchStatus("*", 0); for (auto& b : v) { t->SetBranchStatus(b.n, 1); t->SetBranchAddress(b.n, b.p); } }
static void detach(TTree* t) { t->ResetBranchAddresses(); t->SetBranchStatus("*", 1); }

void probe2(const char* fname = "clusters_seeds_island_RUN_REF-0.root_ntuplizer.root", Long64_t nmax = -1) {
  gStyle->SetOptStat(0); gStyle->SetPalette(kBird);
  TFile* f = TFile::Open(fname);
  TTree* th = (TTree*)f->Get("ntp_hit"); TTree* tc = (TTree*)f->Get("ntp_cluster"); TTree* tk = (TTree*)f->Get("ntp_clus_trk");
  TFile* out = TFile::Open("probe2_hists.root", "RECREATE");
  TCanvas* C = new TCanvas("C", "", 1200, 800); C->Print("probe2.pdf[");
  auto page = [&]() { C->Print("probe2.pdf"); C->Clear(); };

  // ================================================================= A. hits
  {
    Float_t ev, layer, phielem, zelem, phibin, zbin, tbin, adc, e, z;
    std::vector<BR> br = {{"event",&ev},{"layer",&layer},{"phielem",&phielem},{"zelem",&zelem},{"phibin",&phibin},{"zbin",&zbin},{"tbin",&tbin},{"adc",&adc},{"e",&e},{"z",&z}};
    attach(th, br);
    Long64_t N = th->GetEntries(); if (nmax > 0 && nmax < N) N = nmax;
    double tmin[2]={1e9,1e9}, tmax[2]={-1e9,-1e9}, zmin[2]={1e9,1e9}, zmax[2]={-1e9,-1e9}, zbmin[2]={1e9,1e9}, zbmax[2]={-1e9,-1e9};
    double demin = 1e9, demax = -1e9; Long64_t nTPC = 0, nE0 = 0, nEltA = 0, nside[2] = {0,0}, nBadSide = 0, nlowL[55] = {0}, nL[55] = {0};
    TProfile *pz[2], *pzb[2]; TH2D *hz[2], *hzb[2], *hzz[2];
    for (int s = 0; s < 2; ++s) {
      pz[s]  = new TProfile(Form("A_pz%d",s),  "", 1100, -110.5, 989.5);
      pzb[s] = new TProfile(Form("A_pzb%d",s), "", 1100, -110.5, 989.5);
      hz[s]  = new TH2D(Form("A_hz%d",s),  Form("hit z vs tbin, side %d;tbin;z [cm]",s), 400, -110, 990, 260, -130, 130);
      hzb[s] = new TH2D(Form("A_hzb%d",s), Form("hit zbin vs tbin, side %d;tbin;zbin",s), 400, -110, 990, 256, -0.5, 1023.5);
      hzz[s] = new TH2D(Form("A_hzz%d",s), Form("hit zbin vs z, side %d;z [cm];zbin",s), 260, -130, 130, 256, -0.5, 1023.5);
    }
    TH2D* hae = new TH2D("A_hae", "hit adc vs e (TPC);e;adc", 300, 0, 6000, 250, 0, 1000);
    TProfile* pae = new TProfile("A_pae", "", 300, 0, 6000);
    TH1D *hlow[3], *hadc[3];
    for (int r = 0; r < 3; ++r) { hlow[r] = new TH1D(Form("A_low%d",r), "", 41, -0.5, 40.5); hadc[r] = new TH1D(Form("A_adc%d",r), Form("adc R%d",r+1), 1024, -0.5, 1023.5); }
    std::set<int> pads[55], events; std::map<int, std::set<int>> zbOfT;
    std::vector<std::tuple<int,int,int,int,int,int,float>> lowEx;

    for (Long64_t i = 0; i < N; ++i) {
      th->GetEntry(i);
      int L = (int)layer; if (L < 7 || L > 54) continue;
      int s = (int)zelem; if (s < 0 || s > 1) { ++nBadSide; continue; }
      ++nTPC; ++nside[s]; ++nL[L]; events.insert((int)ev);
      tmin[s] = std::min(tmin[s], (double)tbin); tmax[s] = std::max(tmax[s], (double)tbin);
      zmin[s] = std::min(zmin[s], (double)z);    zmax[s] = std::max(zmax[s], (double)z);
      zbmin[s] = std::min(zbmin[s], (double)zbin); zbmax[s] = std::max(zbmax[s], (double)zbin);
      double de = e - adc; demin = std::min(demin, de); demax = std::max(demax, de);
      if (e == 0) ++nE0; if (e < adc) ++nEltA;
      pz[s]->Fill(tbin, z); pzb[s]->Fill(tbin, zbin); hz[s]->Fill(tbin, z); hzb[s]->Fill(tbin, zbin); hzz[s]->Fill(z, zbin);
      hae->Fill(e, adc); pae->Fill(e, adc);
      int r = reg(L); hadc[r]->Fill(adc); if (adc <= 40) hlow[r]->Fill(adc);
      if (adc < 21) { ++nlowL[L]; if (lowEx.size() < 15) lowEx.emplace_back(L, (int)phielem, s, (int)phibin, (int)tbin, (int)adc, e); }
      pads[L].insert((int)phibin);
      auto& S = zbOfT[s * 10000 + (int)tbin + 200]; if (S.size() < 6) S.insert((int)zbin);
    }
    detach(th);

    printf("\n[A] ntp_hit, TPC layers 7..54: %lld of %lld entries scanned -> %lld TPC hits (%.1f%%), side0 %lld side1 %lld, TPC entries with zelem not in {0,1}: %lld\n",
           N, th->GetEntries(), nTPC, 100.*nTPC/N, nside[0], nside[1], nBadSide);
    printf("  events %zu -> %.0f TPC hits/event, %.1f hits/hitset (1152 hitsets)\n", events.size(), (double)nTPC/events.size(), (double)nTPC/events.size()/1152.);
    for (int s = 0; s < 2; ++s) printf("  side %d : tbin [%g, %g]   z [%.2f, %.2f]   zbin [%g, %g]\n", s, tmin[s], tmax[s], zmin[s], zmax[s], zbmin[s], zbmax[s]);
    printf("  e-adc in [%g, %g]   e==0: %lld   e<adc: %lld\n", demin, demax, nE0, nEltA);
    pae->Fit("pol1", "Q0", "", 300, 5000); TF1* fa = pae->GetFunction("pol1");
    { double rms = 0; int nb = 0; for (int b = 1; b <= pae->GetNbinsX(); ++b) if (pae->GetBinEntries(b) > 50 && pae->GetBinCenter(b) > 300 && pae->GetBinCenter(b) < 5000) { rms += pae->GetBinError(b) * pae->GetBinError(b) * pae->GetBinEntries(b); nb += pae->GetBinEntries(b); }
      printf("  A0 adc = %.3f + %.5f*e  (300<e<5000)  -> 1/slope = %.1f e per ADC ; RMS(adc | e) = %.2f ADC (= digitiser noise if flat)\n", fa->GetParameter(0), fa->GetParameter(1), 1./fa->GetParameter(1), nb ? std::sqrt(rms/nb) : -1); }
    printf("  A1 hit z vs tbin:\n"); for (int s = 0; s < 2; ++s) fitline(pz[s], Form("side %d", s));
    printf("  A2 zbin vs tbin:\n");
    for (int s = 0; s < 2; ++s) {
      fitline(pzb[s], Form("side %d", s));
      int multi = 0, tot = 0; for (auto& kv : zbOfT) if (kv.first / 10000 == s) { ++tot; if (kv.second.size() > 1) ++multi; }
      printf("     side %d : %d tbin values, %d with >1 distinct zbin -> %s\n", s, tot, multi, multi == 0 ? "zbin IS a function of (side,tbin)" : "zbin is NOT a function of tbin");
    }
    printf("  A3 hits with adc<21 per layer:"); for (int L = 7; L <= 54; ++L) if (nlowL[L]) printf(" L%d:%lld/%lld", L, nlowL[L], nL[L]); printf("\n     examples (layer sector side phibin tbin adc e):\n");
    for (auto& t : lowEx) printf("       %2d %2d %d %4d %4d %3d %g\n", std::get<0>(t), std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t), std::get<5>(t), std::get<6>(t));
    printf("  A4 adc spectrum 0..40 per region:\n");
    for (int r = 0; r < 3; ++r) { printf("     R%d:", r+1); for (int b = 1; b <= 41; ++b) printf(" %d:%.0f", b-1, hlow[r]->GetBinContent(b));
      int lb = lastBin(hadc[r]); printf("  | n=%.0f  adc_max=%d with %.0f hits (%.3f%%)\n", hadc[r]->GetEntries(), lb-1, hadc[r]->GetBinContent(lb), 100.*hadc[r]->GetBinContent(lb)/hadc[r]->GetEntries()); }
    printf("  A5 distinct phibin per layer (1152/1536/2304 if all pads populated):\n     ");
    for (int L = 7; L <= 54; ++L) printf("L%d:%zu ", L, pads[L].size()); printf("\n");
    for (int L : {7, 22, 23, 39}) { int nexp = L < 23 ? 1152 : L < 39 ? 1536 : 2304, npr = 0; printf("     layer %d missing phibins:", L);
      for (int p = 0; p < nexp; ++p) if (!pads[L].count(p)) { if (npr++ < 40) printf(" %d", p); } printf("  (%d missing of %d)\n", npr, nexp); }

    C->Divide(2, 2); for (int s = 0; s < 2; ++s) { C->cd(1+s); hz[s]->Draw("colz"); gPad->SetLogz(); C->cd(3+s); hzb[s]->Draw("colz"); gPad->SetLogz(); } page();
    C->Divide(2, 2); for (int s = 0; s < 2; ++s) { C->cd(1+s); hzz[s]->Draw("colz"); gPad->SetLogz(); }
    C->cd(3); hae->Draw("colz"); gPad->SetLogz();
    C->cd(4); gPad->SetLogy(); int col[3] = {kRed, kGreen+2, kBlue}; for (int r = 0; r < 3; ++r) { hadc[r]->SetLineColor(col[r]); hadc[r]->GetXaxis()->SetRangeUser(0, 200); hadc[r]->Draw(r ? "same" : ""); } page();
  }

  // ================================================================= B. clusters
  {
    Float_t ev, layer, zelem, phibin, tbin, adc, maxadc, size, phisize, zsize, pedge, x, y, z, ez, ephi;
    std::vector<BR> br = {{"event",&ev},{"layer",&layer},{"zelem",&zelem},{"phibin",&phibin},{"tbin",&tbin},{"adc",&adc},{"maxadc",&maxadc},{"size",&size},{"phisize",&phisize},{"zsize",&zsize},{"pedge",&pedge},{"x",&x},{"y",&y},{"z",&z},{"ez",&ez},{"ephi",&ephi}};
    attach(tc, br);
    Long64_t nTPC = 0, nR[3] = {0}, n1[3] = {0}, n2[3] = {0}, n3[3] = {0}, nps1[3] = {0}, nzs1[3] = {0}, nbump[3] = {0}, nbumpL[55] = {0}, nneg = 0, nbox = 0, dupTPC = 0, dupOther = 0, nside[2] = {0}, nPedge = 0, nSat = 0;
    double amin = 1e9, mamin = 1e9, psmax = 0, zsmax = 0, zsum[2] = {0}, ezs[3] = {0}, ephis[3] = {0}, bumpPs = 0, bumpAdc = 0, bumpPt = 0;
    TH1D* hpedge = new TH1D("B_pedge", "pedge (TPC)", 41, -0.5, 40.5);
    TH2D* hzsL = new TH2D("B_zsL", "zsize vs layer (TPC);layer;zsize", 48, 6.5, 54.5, 40, -0.5, 39.5);
    TH2D* hzsT = new TH2D("B_zsT", "zsize vs tbin (TPC);tbin;zsize", 130, -10, 250, 40, -0.5, 39.5);
    TH2D* hsz = new TH2D("B_sz", "phisize*zsize vs size (TPC);size;phisize*zsize", 130, -130, 130, 130, 0, 130);
    TH1D* hadc = new TH1D("B_adc", "cluster adc (TPC)", 400, 0, 4000);
    TProfile* pz[2]; TH2D* hz[2];
    for (int s = 0; s < 2; ++s) { pz[s] = new TProfile(Form("B_pz%d", s), "", 1100, -110.5, 989.5); hz[s] = new TH2D(Form("B_hz%d", s), Form("cluster z vs tbin, side %d;tbin;z [cm]", s), 400, -110, 990, 260, -130, 130); }
    std::set<std::tuple<int, long, long, long>> seen;
    for (Long64_t i = 0; i < tc->GetEntries(); ++i) {
      tc->GetEntry(i);
      bool dup = !seen.insert(std::make_tuple((int)ev, lrintf(x*1e4f), lrintf(y*1e4f), lrintf(z*1e4f))).second;
      int L = (int)layer, r = reg(L);
      if (r < 0) { if (dup) ++dupOther; continue; }
      if (dup) ++dupTPC;
      ++nTPC; ++nR[r];
      int isz = (int)size, ips = (int)phisize, izs = (int)zsize, box = ips * izs;
      if (isz < 0) ++nneg; if ((int)(int8_t)box == isz) ++nbox;
      hsz->Fill(isz, box); hadc->Fill(adc);
      if (box == 1) ++n1[r]; else if (box == 2) ++n2[r]; else if (box == 3) ++n3[r];
      if (ips == 1) ++nps1[r]; if (izs == 1) ++nzs1[r];
      psmax = std::max(psmax, (double)ips); zsmax = std::max(zsmax, (double)izs);
      if (izs >= 18 && izs <= 26) { ++nbump[r]; ++nbumpL[L]; bumpPs += ips; bumpAdc += adc; bumpPt += tbin; }
      hzsL->Fill(L, izs); hzsT->Fill(tbin, izs); hpedge->Fill(pedge); if (pedge > 0) ++nPedge; if (maxadc >= 963) ++nSat;
      amin = std::min(amin, (double)adc); mamin = std::min(mamin, (double)maxadc);
      ezs[r] += ez; ephis[r] += ephi;
      int s = (int)zelem; if (s == 0 || s == 1) { ++nside[s]; zsum[s] += z; pz[s]->Fill(tbin, z); hz[s]->Fill(tbin, z); }
    }
    detach(tc);
    printf("\n[B] ntp_cluster, TPC only: %lld of %lld entries (R1 %lld R2 %lld R3 %lld), side0 %lld side1 %lld\n", nTPC, tc->GetEntries(), nR[0], nR[1], nR[2], nside[0], nside[1]);
    printf("  size == phisize*zsize (mod int8): %.5f   size<0 (int8 wrap): %lld   max phisize %g  max zsize %g\n", (double)nbox/nTPC, nneg, psmax, zsmax);
    printf("  min cluster adc %g   min maxadc %g   clusters with maxadc>=963 (saturated): %lld (%.3f%%)\n", amin, mamin, nSat, 100.*nSat/nTPC);
    for (int r = 0; r < 3; ++r) printf("  R%d: n=%8lld  bbox=1 %.4f  =2 %.4f  =3 %.4f   phisize==1 %.4f  zsize==1 %.4f   zsize18-26 %.4f   <ez> %.4f <ephi> %.5f\n",
                                      r+1, nR[r], (double)n1[r]/nR[r], (double)n2[r]/nR[r], (double)n3[r]/nR[r], (double)nps1[r]/nR[r], (double)nzs1[r]/nR[r], (double)nbump[r]/nR[r], ezs[r]/nR[r], ephis[r]/nR[r]);
    Long64_t nb = nbump[0]+nbump[1]+nbump[2];
    printf("  zsize 18-26 population: %lld clusters, mean phisize %.2f, mean adc %.0f, mean tbin %.1f ; per layer:", nb, bumpPs/std::max(nb,1LL), bumpAdc/std::max(nb,1LL), bumpPt/std::max(nb,1LL));
    for (int L = 7; L <= 54; ++L) printf(" L%d:%lld", L, nbumpL[L]); printf("\n");
    printf("  pedge>0 fraction %.4f ; distribution:", (double)nPedge/nTPC); for (int b = 1; b <= 41; ++b) if (hpedge->GetBinContent(b)) printf(" %d:%.0f", b-1, hpedge->GetBinContent(b)); printf("\n");
    printf("  z sign convention: <z> side0 = %.2f  side1 = %.2f\n", zsum[0]/std::max(nside[0],1LL), zsum[1]/std::max(nside[1],1LL));
    printf("  cluster z vs tbin (this is the convention tracking uses):\n"); for (int s = 0; s < 2; ++s) fitline(pz[s], Form("side %d", s));
    printf("  exact duplicates (event,x,y,z): TPC %lld   non-TPC %lld\n", dupTPC, dupOther);
    C->Divide(2, 2); C->cd(1); hz[0]->Draw("colz"); gPad->SetLogz(); C->cd(2); hz[1]->Draw("colz"); gPad->SetLogz(); C->cd(3); hzsL->Draw("colz"); gPad->SetLogz(); C->cd(4); hzsT->Draw("colz"); gPad->SetLogz(); page();
    C->Divide(2, 1); C->cd(1); hsz->Draw("colz"); gPad->SetLogz(); C->cd(2); gPad->SetLogy(); hadc->Draw(); page();
  }

  // ================================================================= C. clus_trk
  {
    bool hasSide = tk->GetBranch("zelem") != nullptr;
    Float_t layer, zelem = 0, tbin, x, y, z, resphi, resz, sntpc, spt, seta;
    std::vector<BR> br = {{"layer",&layer},{"tbin",&tbin},{"x",&x},{"y",&y},{"z",&z},{"resphi",&resphi},{"resz",&resz},{"sntpc",&sntpc},{"spt",&spt},{"seta",&seta}};
    if (hasSide) br.push_back({"zelem", &zelem});
    attach(tk, br);
    double s_rp2[3] = {0}, s_rpr2[3] = {0}, s_rz2[3] = {0}; Long64_t n[3] = {0}, nclean = 0, N = tk->GetEntries();
    TProfile* pz[2]; for (int s = 0; s < 2; ++s) pz[s] = new TProfile(Form("C_pz%d", s), "", 1100, -110.5, 989.5);
    TH2D* hrr = new TH2D("C_rr", "resphi*r vs layer;layer;resphi*r [cm if resphi in rad]", 48, 6.5, 54.5, 200, -0.5, 0.5);
    for (Long64_t i = 0; i < N; ++i) {
      tk->GetEntry(i); int r = reg((int)layer); if (r < 0) continue;
      double rad = std::hypot(x, y); ++n[r]; s_rp2[r] += resphi*resphi; s_rpr2[r] += resphi*resphi*rad*rad; s_rz2[r] += resz*resz; hrr->Fill(layer, resphi*rad);
      int s = hasSide ? (int)zelem : (z > 0); if (s == 0 || s == 1) pz[s]->Fill(tbin, z);
      if (sntpc >= 30 && spt > 0.15 && spt < 10 && std::fabs(seta) < 1.1 && std::fabs(resz) < 1) ++nclean;
    }
    detach(tk);
    printf("\n[C] ntp_clus_trk (%s)\n", hasSide ? "side from zelem" : "no zelem branch: side from sign(z)");
    for (int r = 0; r < 3; ++r) printf("  R%d: RMS(resphi) = %.5f   RMS(resphi*r) = %.4f cm   RMS(resz) = %.4f cm\n", r+1, std::sqrt(s_rp2[r]/n[r]), std::sqrt(s_rpr2[r]/n[r]), std::sqrt(s_rz2[r]/n[r]));
    printf("  -> if RMS(resphi*r) ~ 0.1-0.2 cm and grows with r, resphi is in radians; if RMS(resphi) ~ 0.003 is in cm it would be 30 um (unphysical)\n");
    printf("  on-track cluster z vs tbin:\n"); for (int s = 0; s < 2; ++s) fitline(pz[s], Form("side %d", s));
    printf("  clean-track fraction (sntpc>=30, 0.15<spt<10, |seta|<1.1, |resz|<1): %.4f of %lld\n", (double)nclean/N, N);
    hrr->Draw("colz"); gPad->SetLogz(); page();
  }

  C->Print("probe2.pdf]"); out->Write(); out->Close();
  printf("\nwrote probe2.pdf, probe2_hists.root\n");
}