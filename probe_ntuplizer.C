// probe_ntuplizer.C  --  one-shot probe of the sPHENIX TPC ntuplizer output
//
//   root -l -b -q 'probe_ntuplizer.C("clusters_seeds_island_RUN_REF-0.root_ntuplizer.root")'
//   root -l -b -q 'probe_ntuplizer.C("file.root", 2000000)'   // use only 2M ntp_hit entries for heavy probes
//
// Output: stdout report, probe_ntuplizer.pdf (pages), probe_ntuplizer_hists.root (all histograms)

#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TProfile.h"
#include "TF1.h"
#include "TString.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TStyle.h"
#include "TMath.h"
#include <set>
#include <map>
#include <tuple>
#include <cstdio>
#include <cmath>

namespace {
TCanvas* C = nullptr;
TString  PDF = "probe_ntuplizer.pdf";
Long64_t NSUB = 5000000;           // ntp_hit entries used for 2D / spectrum probes (full tree for min/max & sorting)
int      gHid = 0;
const int LREF[6] = {7, 22, 23, 38, 39, 54};      // first/last layer of R1, R2, R3
const char* REG[3] = {"layer>=7&&layer<=22", "layer>=23&&layer<=38", "layer>=39&&layer<=54"};

TString hn(const char* b) { return Form("%s_%d", b, gHid++); }
void hdr(const char* s)   { printf("\n==================== %s ====================\n", s); fflush(stdout); }
void page()               { C->Update(); C->Print(PDF); C->Clear(); C->SetLogz(0); C->SetLogy(0); }

TH1D* h1(TTree* t, const char* var, const char* cut, int nb, double lo, double hi, Long64_t n = TTree::kMaxEntries) {
  TH1D* h = new TH1D(hn("h"), Form("%s  {%s}", var, cut), nb, lo, hi);
  t->Draw(Form("%s>>%s", var, h->GetName()), cut, "goff", n);
  return h;
}
TH2D* h2(TTree* t, const char* var, const char* cut, int nx, double xlo, double xhi, int ny, double ylo, double yhi,
         Long64_t n = TTree::kMaxEntries) {
  TH2D* h = new TH2D(hn("h2"), Form("%s  {%s}", var, cut), nx, xlo, xhi, ny, ylo, yhi);
  t->Draw(Form("%s>>%s", var, h->GetName()), cut, "goff", n);
  return h;
}
TProfile* prof(TTree* t, const char* var, const char* cut, int nb, double lo, double hi, Long64_t n = TTree::kMaxEntries) {
  TProfile* p = new TProfile(hn("p"), Form("%s  {%s}", var, cut), nb, lo, hi);
  t->Draw(Form("%s>>%s", var, p->GetName()), cut, "prof goff", n);
  return p;
}
int firstBin(TH1* h) { for (int b = 1; b <= h->GetNbinsX(); ++b) if (h->GetBinContent(b) > 0) return b; return -1; }
int lastBin(TH1* h)  { for (int b = h->GetNbinsX(); b >= 1; --b) if (h->GetBinContent(b) > 0) return b; return -1; }

void minmax(TTree* t, const char* list) {
  TObjArray* a = TString(list).Tokenize(" ");
  printf("  %-14s %16s %16s   note\n", "branch", "min", "max");
  for (int i = 0; i < a->GetEntriesFast(); ++i) {
    const char* b = ((TObjString*)a->At(i))->GetString().Data();
    double lo = t->GetMinimum(b), hi = t->GetMaximum(b);
    printf("  %-14s %16.7g %16.7g   %s\n", b, lo, hi, lo == hi ? "CONSTANT" : "");
  }
  delete a;
}
void quant(TH1* h, const char* label) {
  const int N = 7; double p[N] = {0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99}, q[N];
  h->GetQuantiles(N, q, p);
  printf("  %-10s n=%-9.0f under=%-7.0f over=%-7.0f q01=%-9.4g q05=%-9.4g q16=%-9.4g med=%-9.4g q84=%-9.4g q95=%-9.4g q99=%-9.4g\n",
         label, h->GetEntries(), h->GetBinContent(0), h->GetBinContent(h->GetNbinsX() + 1),
         q[0], q[1], q[2], q[3], q[4], q[5], q[6]);
}
}  // namespace

void probe_ntuplizer(const char* fname = "clusters_seeds_island_RUN_REF-0.root_ntuplizer.root", Long64_t nsub = 5000000) {
  NSUB = nsub;
  gStyle->SetOptStat(1110); gStyle->SetPalette(kBird); gStyle->SetNumberContours(64);
  C = new TCanvas("C", "probe", 1000, 750);
  C->Print(PDF + "[");

  TFile* f = TFile::Open(fname);
  if (!f || f->IsZombie()) { printf("cannot open %s\n", fname); return; }
  TTree* th = (TTree*)f->Get("ntp_hit");
  TTree* tc = (TTree*)f->Get("ntp_cluster");
  TTree* tk = (TTree*)f->Get("ntp_clus_trk");
  TTree* ti = (TTree*)f->Get("ntp_info");
  TFile* out = TFile::Open("probe_ntuplizer_hists.root", "RECREATE");   // histograms land here

  // ------------------------------------------------------------------ [0]
  hdr("[0] file / trees");
  printf("  %s\n", fname);
  printf("  ntp_info     %12lld entries (events)\n", ti ? ti->GetEntries() : -1);
  printf("  ntp_hit      %12lld entries\n", th->GetEntries());
  printf("  ntp_cluster  %12lld entries\n", tc->GetEntries());
  printf("  ntp_clus_trk %12lld entries   (%.2f %% of ntp_cluster)\n", tk->GetEntries(),
         100. * tk->GetEntries() / tc->GetEntries());
  printf("  heavy ntp_hit probes use the first %lld entries\n", NSUB);

  // ------------------------------------------------------------------ [1]
  if (ti) {
    hdr("[1] ntp_info : branches + first 10 events");
    TObjArray* br = ti->GetListOfBranches();
    printf("  branches:"); for (int i = 0; i < br->GetEntries(); ++i) printf(" %s", br->At(i)->GetName()); printf("\n");
    ti->Scan("*", "", "colsize=9", 10);
    for (const char* v : {"nhittpcall", "nclustpc", "ntpcseed", "ntrk", "occ11", "occ116", "occ21", "occ216", "occ31", "occ316"})
      if (ti->GetBranch(v)) {
        TH1D* h = h1(ti, v, "", 1000, ti->GetMinimum(v) - 0.5, ti->GetMaximum(v) + 0.5);
        printf("  %-11s mean %14.4g   min %12.6g  max %12.6g  sum %14.6g\n", v, h->GetMean(), ti->GetMinimum(v), ti->GetMaximum(v), h->GetMean() * h->GetEntries());
      }
    printf("  -> compare sum(nhittpcall) with ntp_hit entries = %lld (are all hits written, or only some events?)\n", th->GetEntries());
  }

  // ------------------------------------------------------------------ [2]
  hdr("[2a] ntp_hit : first 25 entries");
  th->Scan("event:layer:phielem:zelem:phibin:zbin:tbin:adc:e:ecell:seed:cellID:z:r:phi", "", "colsize=8", 25);

  hdr("[2b] ntp_hit : min/max over FULL tree (CONSTANT = unfilled branch)");
  minmax(th, "event seed run seg job cellID layer phielem zelem phibin zbin tbin adc e ecell z r phi hitID ntrk ntpcseed nclustpc");
  const double tmax = th->GetMaximum("tbin");
  const double evmax = th->GetMaximum("event");

  hdr("[2c] ntp_hit : is e == adc ? is ecell == adc ?");
  printf("  e-adc      min %g max %g\n", th->GetMinimum("e") - 0, 0.);  // placeholder replaced below
  {
    TH1D* h = h1(th, "e-adc", "", 2001, -1000.5, 1000.5, NSUB);
    TH1D* g = h1(th, "ecell-adc", "", 2001, -1000.5, 1000.5, NSUB);
    printf("  e-adc      : bins occupied [%g, %g]  -> %s\n", h->GetBinCenter(firstBin(h)), h->GetBinCenter(lastBin(h)), firstBin(h) == lastBin(h) ? "IDENTICAL (drop e)" : "differ");
    printf("  ecell-adc  : bins occupied [%g, %g]  -> %s\n", g->GetBinCenter(firstBin(g)), g->GetBinCenter(lastBin(g)), firstBin(g) == lastBin(g) ? "IDENTICAL (drop ecell)" : "differ");
  }

  hdr("[2d] ntp_hit : phibin numbering  (global per layer: max ~1151/1535/2303 ; local per sector: max ~95/127/191)");
  for (int L : LREF) {
    TH1D* h = h1(th, "phibin", Form("layer==%d", L), 4096, -0.5, 4095.5, NSUB);
    int lo = firstBin(h) - 1, hi = lastBin(h) - 1;
    printf("  layer %2d : phibin in [%4d, %4d]  n=%.0f   -> %s\n", L, lo, hi, h->GetEntries(), hi > 300 ? "GLOBAL" : "LOCAL (per sector)");
  }
  { TH2D* h = h2(th, "phibin:phielem", "layer==7", 12, -0.5, 11.5, 300, -0.5, 1199.5, NSUB); h->Draw("colz"); C->SetLogz(); page(); }
  { TH1D* h = h1(th, "phielem", "", 12, -0.5, 11.5, NSUB); TH1D* g = h1(th, "zelem", "", 2, -0.5, 1.5, NSUB);
    C->Divide(2, 1); C->cd(1); h->Draw(); C->cd(2); g->Draw(); page(); }

  hdr("[2e] ntp_hit : zbin vs tbin  (is zbin a deterministic function of tbin & side?)");
  th->Scan("zelem:tbin:zbin:z", "", "colsize=8", 12);
  for (int s = 0; s < 2; ++s) {
    TProfile* p = prof(th, "zbin:tbin", Form("zelem==%d", s), (int)tmax + 1, -0.5, tmax + 0.5, NSUB);
    p->Fit("pol1", "Q0");
    TF1* fn = p->GetFunction("pol1");
    // spread of zbin at fixed tbin: if ~0, deterministic
    TH2D* h = h2(th, "zbin:tbin", Form("zelem==%d", s), (int)tmax + 1, -0.5, tmax + 0.5, 600, -0.5, 599.5, NSUB);
    double maxrms = 0; for (int b = 1; b <= h->GetNbinsX(); ++b) { TH1D* py = h->ProjectionY("_py", b, b); if (py->GetEntries() > 10) maxrms = TMath::Max(maxrms, py->GetRMS()); delete py; }
    printf("  side %d : zbin = %.4f + %.5f*tbin   max RMS(zbin | tbin) = %.3f  -> %s\n", s, fn->GetParameter(0), fn->GetParameter(1), maxrms,
           maxrms < 0.6 ? "deterministic (zbin is a relabelled tbin)" : "NOT a pure function of tbin");
    h->Draw("colz"); C->SetLogz(); page();
  }

  hdr("[2f] ntp_hit : ADC spectrum per region -> zero-suppression threshold (first populated ADC value)");
  C->Divide(3, 1);
  for (int r = 0; r < 3; ++r) {
    TH1D* h = h1(th, "adc", REG[r], 1024, -0.5, 1023.5, NSUB);
    int thr = firstBin(h) - 1;
    double n = h->GetEntries();
    printf("  R%d %-24s : adc_min=%d  adc_max=%d  frac(adc<=thr+2)=%.3f  frac(adc<=thr+5)=%.3f  frac(adc<=thr+10)=%.3f  frac(adc>=1000)=%.4f\n",
           r + 1, REG[r], thr, lastBin(h) - 1, h->Integral(1, thr + 3) / n, h->Integral(1, thr + 6) / n, h->Integral(1, thr + 11) / n, h->Integral(1001, 1024) / n);
    C->cd(r + 1); gPad->SetLogy(); h->GetXaxis()->SetRangeUser(0, 300); h->Draw();
  }
  page();

  hdr("[2g] ntp_hit : z vs tbin per side -> calibration  z = z0 + slope*tbin   (from the file's own z column)");
  C->Divide(2, 1);
  for (int s = 0; s < 2; ++s) {
    TProfile* p = prof(th, "z:tbin", Form("zelem==%d", s), (int)tmax + 1, -0.5, tmax + 0.5, NSUB);
    p->Fit("pol1", "Q0");
    TF1* fn = p->GetFunction("pol1");
    double zlo = 1e9, zhi = -1e9;
    { TH1D* hz = h1(th, "z", Form("zelem==%d", s), 2400, -120, 120, NSUB); zlo = hz->GetBinLowEdge(firstBin(hz)); zhi = hz->GetBinLowEdge(lastBin(hz) + 1); }
    double rms = 0; int nb = 0;
    for (int b = 1; b <= p->GetNbinsX(); ++b) if (p->GetBinEntries(b) > 10) { rms += p->GetBinError(b) * p->GetBinError(b) * p->GetBinEntries(b); ++nb; }
    printf("  side %d : z0 = %10.4f cm   slope = %9.5f cm/tbin   z range [%.2f, %.2f]   tbin range used [%d, %d]  |z| at tbin=0 -> %s\n",
           s, fn->GetParameter(0), fn->GetParameter(1), zlo, zhi, firstBin(p) - 1, lastBin(p) - 1,
           std::fabs(fn->GetParameter(0)) > 50 ? "readout plane (drift grows with tbin toward CM)" : "central membrane (drift grows away from CM)");
    C->cd(s + 1); TH2D* h = h2(th, "z:tbin", Form("zelem==%d", s), (int)tmax + 1, -0.5, tmax + 0.5, 240, -120, 120, NSUB); h->Draw("colz"); gPad->SetLogz();
  }
  page();

  hdr("[2h] ntp_hit : r per layer (geometry table), hits per layer");
  {
    TProfile* p = prof(th, "r:layer", "", 60, -0.5, 59.5, NSUB);
    printf("  layer : <r> cm  (only TPC layers 7..54 expected)\n  ");
    for (int b = 1; b <= 60; ++b) if (p->GetBinEntries(b) > 0) printf("%d:%.2f ", b - 1, p->GetBinContent(b));
    printf("\n");
    TH1D* h = h1(th, "layer", "", 60, -0.5, 59.5, NSUB);
    printf("  hits per layer (fraction): R1 %.3f  R2 %.3f  R3 %.3f\n", h->Integral(8, 23) / h->GetEntries(), h->Integral(24, 39) / h->GetEntries(), h->Integral(40, 55) / h->GetEntries());
    h->Draw(); page();
  }

  hdr("[2i] ntp_hit : event ordering, distinct events, hits/event");
  {
    th->SetBranchStatus("*", 0); th->SetBranchStatus("event", 1);
    Float_t ev; th->SetBranchAddress("event", &ev);
    Long64_t n = th->GetEntries(), nbreak = 0; float prev = -1e30f; std::map<int, Long64_t> cnt;
    for (Long64_t i = 0; i < n; ++i) { th->GetEntry(i); if (ev < prev) ++nbreak; prev = ev; ++cnt[(int)ev]; }
    th->ResetBranchAddresses(); th->SetBranchStatus("*", 1);
    Long64_t mn = 1LL << 60, mx = 0; for (auto& kv : cnt) { mn = TMath::Min(mn, kv.second); mx = TMath::Max(mx, kv.second); }
    printf("  distinct events in ntp_hit: %zu   (ntp_info says %lld)   event id range [%d, %d]\n", cnt.size(), ti ? ti->GetEntries() : -1, cnt.begin()->first, cnt.rbegin()->first);
    printf("  ordering breaks (event decreases): %lld  -> %s\n", nbreak, nbreak == 0 ? "SORTED by event (chunked reading is safe)" : "NOT sorted");
    printf("  hits/event: mean %.0f  min %lld  max %lld\n", (double)n / cnt.size(), mn, mx);
    printf("  first 5 events:"); int k = 0; for (auto& kv : cnt) { if (k++ >= 5) break; printf("  ev%d:%lld", kv.first, kv.second); } printf("\n");
    printf("  mean hits per hitset (layer x sector x side = 48*24 = 1152 per event): %.1f\n", (double)n / cnt.size() / 1152.);
    TH1D* h = h1(th, "event", "", (int)evmax + 1, -0.5, evmax + 0.5); h->SetTitle("hits per event"); h->Draw("hist"); page();
  }

  // ------------------------------------------------------------------ [3]
  hdr("[3a] ntp_cluster : first 15 entries");
  tc->Scan("event:layer:phielem:zelem:phibin:tbin:adc:maxadc:size:phisize:zsize:pedge:fee:chan:sampa:ez:ephi:x:y:z", "", "colsize=8", 15);

  hdr("[3b] ntp_cluster : min/max (full tree)");
  minmax(tc, "event seed layer phielem zelem phibin tbin adc maxadc size phisize zsize pedge redge ovlp trackID niter fee chan sampa ex ey ez ephi pez pephi e thick afac bfac dcal x y z r");

  hdr("[3c] ntp_cluster : centroid convention  (fractional phibin/tbin ?  same numbering as hits ?)");
  {
    TH1D* hf = h1(tc, "phibin-TMath::Floor(phibin)", "", 100, 0, 1);
    TH1D* tf = h1(tc, "tbin-TMath::Floor(tbin)", "", 100, 0, 1);
    printf("  fraction of clusters with integer phibin: %.3f   integer tbin: %.3f   -> %s\n",
           hf->GetBinContent(1) / hf->GetEntries(), tf->GetBinContent(1) / tf->GetEntries(),
           hf->GetBinContent(1) / hf->GetEntries() < 0.5 ? "float centroids in bin units (compare to my cog_phi/cog_t)" : "INTEGER (bin index, not centroid)");
    for (int L : LREF) {
      TH1D* h = h1(tc, "phibin", Form("layer==%d", L), 4096, -0.5, 4095.5);
      printf("  layer %2d : cluster phibin in [%4d, %4d]  n=%.0f\n", L, firstBin(h) - 1, lastBin(h) - 1, h->GetEntries());
    }
    printf("  -> must agree with [2d] or the hit<->cluster matching needs an offset of phielem*npad_sector\n");
    TH1D* he = h1(tc, "e-adc", "", 2001, -1000.5, 1000.5);
    printf("  cluster e-adc occupied [%g, %g] -> %s\n", he->GetBinCenter(firstBin(he)), he->GetBinCenter(lastBin(he)), firstBin(he) == lastBin(he) ? "IDENTICAL" : "differ");
  }

  hdr("[3d] ntp_cluster : size / shape / edge / electronics");
  {
    TH1D* hs = h1(tc, "size", "", 60, -0.5, 59.5), *hp = h1(tc, "phisize", "", 20, -0.5, 19.5), *hz = h1(tc, "zsize", "", 40, -0.5, 39.5);
    double n = hs->GetEntries();
    printf("  size   : frac(1)=%.3f frac(<=2)=%.3f frac(<=3)=%.3f  mean %.2f  overflow(>59)=%.0f\n", hs->GetBinContent(1) / n, hs->Integral(1, 2) / n, hs->Integral(1, 3) / n, hs->GetMean(), hs->GetBinContent(61));
    printf("  phisize: frac(1)=%.3f  mean %.2f      zsize: frac(1)=%.3f  mean %.2f\n", hp->GetBinContent(1) / n, hp->GetMean(), hz->GetBinContent(1) / n, hz->GetMean());
    quant(hs, "size"); quant(hp, "phisize"); quant(hz, "zsize");
    TH1D* ha = h1(tc, "adc", "", 4000, 0, 20000), *hm = h1(tc, "maxadc", "", 1100, -0.5, 1099.5);
    quant(ha, "adc"); quant(hm, "maxadc");
    TH1D* hpe = h1(tc, "pedge", "", 2, -0.5, 1.5);
    printf("  pedge==1 fraction: %.4f\n", hpe->GetBinContent(2) / n);
    TH1D* hfee = h1(tc, "fee", "", 30, -0.5, 29.5), *hch = h1(tc, "chan", "", 260, -0.5, 259.5), *hsa = h1(tc, "sampa", "", 10, -0.5, 9.5);
    printf("  fee range [%d,%d]  chan range [%d,%d]  sampa range [%d,%d]\n", firstBin(hfee) - 1, lastBin(hfee) - 1, firstBin(hch) - 1, lastBin(hch) - 1, firstBin(hsa) - 1, lastBin(hsa) - 1);
    TH1D* hev = h1(tc, "event", "", (int)evmax + 1, -0.5, evmax + 0.5);
    printf("  clusters/event mean %.0f\n", tc->GetEntries() / (double)std::max(1, (int)(hev->GetEntries() > 0 ? (lastBin(hev) - firstBin(hev) + 1) : 1)));
    TH1D* hl = h1(tc, "layer", "", 60, -0.5, 59.5);
    printf("  clusters per layer (fraction): R1 %.3f  R2 %.3f  R3 %.3f\n", hl->Integral(8, 23) / n, hl->Integral(24, 39) / n, hl->Integral(40, 55) / n);
    C->Divide(3, 2);
    C->cd(1); gPad->SetLogy(); hs->Draw(); C->cd(2); gPad->SetLogy(); hp->Draw(); C->cd(3); gPad->SetLogy(); hz->Draw();
    C->cd(4); gPad->SetLogy(); ha->GetXaxis()->SetRangeUser(0, 3000); ha->Draw(); C->cd(5); gPad->SetLogy(); hm->Draw(); C->cd(6); hl->Draw();
    page();
    TH2D* h2s = h2(tc, "zsize:phisize", "", 12, -0.5, 11.5, 30, -0.5, 29.5); h2s->Draw("colz text"); C->SetLogz(); page();
    TH2D* h2a = h2(tc, "adc:size", "", 40, -0.5, 39.5, 200, 0, 4000); h2a->Draw("colz"); C->SetLogz(); page();
    TH2D* h2z = h2(tc, "ez:layer", "", 60, -0.5, 59.5, 200, 0, 2); h2z->Draw("colz"); C->SetLogz(); page();
  }

  // ------------------------------------------------------------------ [4]
  hdr("[4a] ntp_clus_trk : first 15 entries");
  tk->Scan("event:seedID:layer:phibin:tbin:size:sntpc:snsil:snhits:spt:seta:scharge:sdedx:alpha:beta:resphio:resphi:resz", "", "colsize=8", 15);

  hdr("[4b] ntp_clus_trk : min/max (full tree)");
  minmax(tk, "event seedID siter seed layer size sntpc snsil snhits sn1pix spt sptot seta sphi scharge sdedx spidedx skdedx sprdedx alpha beta resphio resphi resz sX0 sY0 sdZ0 sR0 afac bfac thick");

  hdr("[4c] ntp_clus_trk : seed / residual quantiles  (-> numbers for the on-track 'clean seed' cut)");
  {
    quant(h1(tk, "sntpc", "", 80, -0.5, 79.5), "sntpc");
    quant(h1(tk, "spt", "", 2000, 0, 20), "spt");
    quant(h1(tk, "seta", "", 400, -2, 2), "seta");
    quant(h1(tk, "sdedx", "", 2000, 0, 20000), "sdedx");
    quant(h1(tk, "alpha", "", 800, -4, 4), "alpha");
    quant(h1(tk, "beta", "", 800, -4, 4), "beta");
    quant(h1(tk, "resphi", "", 4000, -2, 2), "resphi");
    quant(h1(tk, "resphio", "", 4000, -2, 2), "resphio");
    quant(h1(tk, "resz", "", 4000, -5, 5), "resz");
    quant(h1(tk, "size", "", 60, -0.5, 59.5), "size(trk)");
    printf("  (compare size(trk) with [3d] size: on-track clusters should be larger)\n");
    // seeds per event
    std::set<std::pair<int, int>> seeds; Float_t ev, sid;
    tk->SetBranchStatus("*", 0); tk->SetBranchStatus("event", 1); tk->SetBranchStatus("seedID", 1);
    tk->SetBranchAddress("event", &ev); tk->SetBranchAddress("seedID", &sid);
    for (Long64_t i = 0; i < tk->GetEntries(); ++i) { tk->GetEntry(i); seeds.insert({(int)ev, (int)sid}); }
    tk->ResetBranchAddresses(); tk->SetBranchStatus("*", 1);
    printf("  distinct (event,seedID): %zu  -> %.1f clusters per seed, %.1f seeds per event\n", seeds.size(), (double)tk->GetEntries() / seeds.size(),
           (double)seeds.size() / (ti ? ti->GetEntries() : 1));
    C->Divide(3, 2);
    C->cd(1); h1(tk, "resphi", "", 200, -0.5, 0.5)->Draw(); C->cd(2); h1(tk, "resz", "", 200, -2, 2)->Draw();
    C->cd(3); h1(tk, "sntpc", "", 60, -0.5, 59.5)->Draw(); C->cd(4); gPad->SetLogy(); h1(tk, "spt", "", 200, 0, 10)->Draw();
    C->cd(5); h1(tk, "alpha", "", 200, -2, 2)->Draw(); C->cd(6); h1(tk, "beta", "", 200, -2, 2)->Draw();
    page();
    { TH2D* h = h2(tk, "sdedx:spt*scharge", "", 200, -4, 4, 200, 0, 8000); h->Draw("colz"); C->SetLogz(); page(); }
    { TH2D* h = h2(tk, "resphi:alpha", "", 100, -2, 2, 100, -0.3, 0.3); h->Draw("colz"); C->SetLogz(); page(); }
    { TH2D* h = h2(tk, "size:layer", "", 48, 6.5, 54.5, 30, -0.5, 29.5); h->Draw("colz"); C->SetLogz(); page(); }
  }

  // ------------------------------------------------------------------ [5]
  hdr("[5] cross-tree : is ntp_clus_trk a subset of ntp_cluster ?  (match on event + x,y,z rounded to 1 um)");
  {
    typedef std::tuple<int, long, long, long> K;
    auto key = [](float ev, float x, float y, float z) { return K((int)ev, lrintf(x * 1e4f), lrintf(y * 1e4f), lrintf(z * 1e4f)); };
    std::set<K> S; Float_t ev, x, y, z;
    for (TTree* t : {tc, tk}) {
      t->SetBranchStatus("*", 0); for (const char* b : {"event", "x", "y", "z"}) t->SetBranchStatus(b, 1);
      t->SetBranchAddress("event", &ev); t->SetBranchAddress("x", &x); t->SetBranchAddress("y", &y); t->SetBranchAddress("z", &z);
    }
    for (Long64_t i = 0; i < tc->GetEntries(); ++i) { tc->GetEntry(i); S.insert(key(ev, x, y, z)); }
    Long64_t found = 0, n = tk->GetEntries(); std::set<K> T;
    for (Long64_t i = 0; i < n; ++i) { tk->GetEntry(i); K k = key(ev, x, y, z); T.insert(k); if (S.count(k)) ++found; }
    for (TTree* t : {tc, tk}) { t->ResetBranchAddresses(); t->SetBranchStatus("*", 1); }
    printf("  ntp_cluster distinct (event,x,y,z): %zu of %lld  (duplicates: %lld)\n", S.size(), tc->GetEntries(), tc->GetEntries() - (Long64_t)S.size());
    printf("  ntp_clus_trk entries found in ntp_cluster: %lld / %lld = %.4f   distinct clus_trk clusters: %zu (%.1f%% of official clusters are on a seed)\n",
           found, n, (double)found / n, T.size(), 100. * T.size() / S.size());
    printf("  -> if ~1.0: on-track labels can be attached to official clusters by exact (x,y,z) join instead of a tolerance match\n");
  }

  C->Print(PDF + "]");
  out->Write(); out->Close();
  printf("\nwrote %s and probe_ntuplizer_hists.root\n", PDF.Data());
}