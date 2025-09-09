import numpy as np
import awkward as ak
import seaborn as sns
import os
import vector 
import numpy as np
import pandas as pd
import uproot
import sys
import copy


behavior = vector.register_awkward()


def extract_features(events, label, max_leptons=3, max_jets=2):
    
    def safe_get(arr, idx, default=-999):
        padded = ak.pad_none(arr, idx + 1, axis=1)
        extracted = padded[:, idx]
        return ak.to_numpy(ak.fill_none(extracted, default))


    def safe_get_jets(array, idx, max_items):
        padded = ak.pad_none(array, max_items, axis=1)
        extracted = padded[:, idx]
        return ak.to_numpy(extracted, allow_missing=True)

    lep_pairmass = compute_pairwise_masses_3lep(events)
    lep_dvars = compute_delta_eta_phi_3lep(events)
    lep_dR = compute_deltaR_3lep(events)
    #best_mass = compute_best_z_candidate_masses(events, particle_type="mu", z_window=10.0)
    sph, apl = compute_event_shapes(events)
    zfe_mu = z_met_balance_features(events, "mu")
    zfe_el = z_met_balance_features(events, "el")
    #thr = compute_thrust(events, n_steps=100)
    features = {
        #leading leptons
        "el_pt_0": safe_get(events.el_pt, 0, max_leptons),
        "el_eta_0": safe_get(events.el_eta, 0, max_leptons),
        "el_phi_0": safe_get(events.el_phi, 0, max_leptons),

        "mu_pt_0": safe_get(events.mu_pt, 0, max_leptons),
        "mu_eta_0": safe_get(events.mu_eta, 0, max_leptons),
        "mu_phi_0": safe_get(events.mu_phi, 0, max_leptons),

        #2nd leading leptons
        "el_pt_1": safe_get(events.el_pt, 1, max_leptons),
        "el_eta_1": safe_get(events.el_eta, 1, max_leptons),
        "el_phi_1": safe_get(events.el_phi, 1, max_leptons),

        "mu_pt_1": safe_get(events.mu_pt, 1, max_leptons),
        "mu_eta_1": safe_get(events.mu_eta, 1, max_leptons),
        "mu_phi_1": safe_get(events.mu_phi, 1, max_leptons),

        #third leading leptons
        "el_pt_2": safe_get(events.el_pt, 2, max_leptons),
        "el_eta_2": safe_get(events.el_eta, 2, max_leptons),
        "el_phi_2": safe_get(events.el_phi, 2, max_leptons),

        "mu_pt_2": safe_get(events.mu_pt, 2, max_leptons),
        "mu_eta_2": safe_get(events.mu_eta, 2, max_leptons),
        "mu_phi_2": safe_get(events.mu_phi, 2, max_leptons),

        #jets
        "jet_pt_0": safe_get_jets(events.jet_pt, 0, max_jets),
        "jet_eta_0": safe_get_jets(events.jet_eta, 0, max_jets),
        "jet_phi_0": safe_get_jets(events.jet_phi, 0, max_jets),

        "jet_pt_1": safe_get_jets(events.jet_pt, 1, max_jets),
        "jet_eta_1": safe_get_jets(events.jet_eta, 1, max_jets),
        "jet_phi_1": safe_get_jets(events.jet_phi, 1, max_jets),
        "jet_mass_0": safe_get_jets(events.jet_mass, 0, max_jets),
        "jet_btag_0": safe_get_jets(events.jet_btag, 0, max_jets),

        #other/new features
        "MET": ak.to_numpy(events.MET, allow_missing=True),
        "MET_Phi": ak.to_numpy(events.MET_Phi, allow_missing=True),

        "m_3lep" : compute_invariant_mass_3lep(events),
        "total_event_et": total_event_et(events),
        "MT_lep_MET" : compute_mt(events),
        "best_Z_mass_mu" : compute_best_z_candidate_masses(events, particle_type="mu"),
        "best_Z_mass_el" : compute_best_z_candidate_masses(events, particle_type="el"),
        "best_Z_pt_mu": compute_best_z_candidate_pt(events, particle_type="mu"),
        "best_Z_pt_el": compute_best_z_candidate_pt(events, particle_type="el"),

        "del_phi_Z_MET_mu":compute_delta_phi_Z_MET(events, particle_type="mu"),
        "del_phi_Z_MET_el":compute_delta_phi_Z_MET(events, particle_type="el"),

        "del_phi_Z_lep_mu":compute_delta_phi_Z_lep(events, particle_type="mu"),
        "del_phi_Z_lep_el":compute_delta_phi_Z_lep(events, particle_type="el"),



        "pt_3lepsys": compute_pt_3lepsys(events),
        "dphi_met_3lepsys": compute_dphi_met_3lepsys(events),

        #"mt_nonZ_lepton_MET_el": mt_nonZ_lepton_MET(events, particle_type="el"),
        #"mt_nonZ_lepton_MET_mu": mt_nonZ_lepton_MET(events, particle_type="mu"),
        #"Z_MET_dphi_mu": zfe_mu["dphi_Z_MET"],
        "Z_MET_par_mu":  zfe_mu["met_par_to_Z"],
        "Z_MET_perp_mu": zfe_mu["met_perp_to_Z"],
        "Z_MET_ratio_mu": zfe_mu["met_over_pTZ"],
        "Z_MET_diff_mu":  zfe_mu["pTZ_minus_MET"],
        "Z_MET_recoil_mu": zfe_mu["recoil_Z_MET"],

        #"Z_MET_dphi_el": zfe_el["dphi_Z_MET"],
        "Z_MET_par_el":  zfe_el["met_par_to_Z"],
        "Z_MET_perp_el": zfe_el["met_perp_to_Z"],
        "Z_MET_ratio_el": zfe_el["met_over_pTZ"],
        "Z_MET_diff_el":  zfe_el["pTZ_minus_MET"],
        "Z_MET_recoil_el": zfe_el["recoil_Z_MET"],
        
        #lep mass pairs
        "m12": lep_pairmass["m12"],
        "m23": lep_pairmass["m23"],
        "m31": lep_pairmass["m31"],

        #deta, dphi between leps
        "deta12": lep_dvars["deta12"],
        "deta23": lep_dvars["deta23"],
        "deta31": lep_dvars["deta31"],
        "dphi12": lep_dvars["dphi12"],
        "dphi23": lep_dvars["dphi23"],
        "dphi31": lep_dvars["dphi31"],

        #lep del R pairs
        "deltaR12": lep_dR["deltaR12"],
        "deltaR23": lep_dR["deltaR23"],
        "deltaR31": lep_dR["deltaR31"],
        
        "sphericity" : sph, 
        "aplanarity" : apl,
        
        #"thrust" : thr
    }
    #print("MET:", ak.to_numpy(events.MET, allow_missing=True)[:5])
    #print("MET_Phi:", ak.to_numpy(events.MET_Phi, allow_missing=True)[:5])

    print("m_3lep:", compute_invariant_mass_3lep(events)[:5])
    print("MT_lep_MET:", compute_mt(events)[:5])
    print("total_event_et:", total_event_et(events)[:5])
    print("best_Z_mass_mu:", compute_best_z_candidate_masses(events, particle_type="mu")[:5])
    print("best_Z_mass_el:", compute_best_z_candidate_masses(events, particle_type="el")[:5])
    print("best_Z_pt_mu:", compute_best_z_candidate_pt(events, particle_type="mu")[:5])
    print("best_Z_pt_el:", compute_best_z_candidate_pt(events, particle_type="el")[:5])

    print("del_phi_Z_MET_mu:", compute_delta_phi_Z_MET(events, particle_type="mu")[:5])
    print("del_phi_Z_MET_el:", compute_delta_phi_Z_MET(events, particle_type="el")[:5])


    print("pt_3lepsys:", compute_pt_3lepsys(events)[:5])
    print("dphi_met_3lepsys:", compute_dphi_met_3lepsys(events)[:5])

    #print("mt_nonZ_lepton_MET_mu:", mt_nonZ_lepton_MET(events, particle_type="mu")[:5])
    #print("mt_nonZ_lepton_MET_el:", mt_nonZ_lepton_MET(events, particle_type="el")[:5])
    #print("Z_MET_dphi_mu:", zfe_mu["dphi_Z_MET"][:5]),
    print("Z_MET_par_mu:",  zfe_mu["met_par_to_Z"][:5]),
    print("Z_MET_perp_mu:", zfe_mu["met_perp_to_Z"][:5]),
    print("Z_MET_ratio_mu:", zfe_mu["met_over_pTZ"][:5]),
    print("Z_MET_diff_mu:",  zfe_mu["pTZ_minus_MET"][:5]),
    print("Z_MET_recoil_mu:", zfe_mu["recoil_Z_MET"][:5]),

    #print("Z_MET_dphi_el:", zfe_el["dphi_Z_MET"][:5]),
    print("Z_MET_par_el:",  zfe_el["met_par_to_Z"][:5]),
    print("Z_MET_perp_el:", zfe_el["met_perp_to_Z"][:5]),
    print("Z_MET_ratio_el:", zfe_el["met_over_pTZ"][:5]),
    print("Z_MET_diff_el:",  zfe_el["pTZ_minus_MET"][:5]),
    print("Z_MET_recoil_el:", zfe_el["recoil_Z_MET"][:5]),


    
    print("m12:", ak.to_numpy(lep_pairmass["m12"], allow_missing=True)[:5])
    print("m23:", ak.to_numpy(lep_pairmass["m23"], allow_missing=True)[:5])
    print("m31:", ak.to_numpy(lep_pairmass["m31"], allow_missing=True)[:5])

    print("deta12:", ak.to_numpy(lep_dvars["deta12"], allow_missing=True)[:5])
    print("deta23:", ak.to_numpy(lep_dvars["deta23"], allow_missing=True)[:5])
    print("deta31:", ak.to_numpy(lep_dvars["deta31"], allow_missing=True)[:5])
    print("dphi12:", ak.to_numpy(lep_dvars["dphi12"], allow_missing=True)[:5])
    print("dphi23:", ak.to_numpy(lep_dvars["dphi23"], allow_missing=True)[:5])
    print("dphi31:", ak.to_numpy(lep_dvars["dphi31"], allow_missing=True)[:5])

    print("deltaR12:", ak.to_numpy(lep_dR["deltaR12"], allow_missing=True)[:5])
    print("deltaR23:", ak.to_numpy(lep_dR["deltaR23"], allow_missing=True)[:5])
    print("deltaR31:", ak.to_numpy(lep_dR["deltaR31"], allow_missing=True)[:5])  
    
    print("sphericity:", sph[:5])
    print("aplanarity:", apl[:5])

    
    #print("thrust:", thr[:5])
    
    lengths = [len(arr) for arr in features.values()]  #this sanity check ensurs all the arrays have same length, got a lot of errors
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent array lengths in the features, not good: {set(lengths)}")

    df = pd.DataFrame(features)
    df["label"] = label
    return df




def delta_r(eta1, phi1, eta2, phi2):
  
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    deta = eta1 - eta2
    dr2 = deta**2 + dphi**2
    dr2 = np.where(np.isnan(dr2), np.nan, np.maximum(dr2, 0))  #protecting against NaN/negatives
    return np.sqrt(dr2)

#inv mass func
def invariant_mass(pt1, eta1, phi1, mass1, pt2, eta2, phi2, mass2):
    def to_cartesian(pt, eta, phi, mass):
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
        return px, py, pz, energy

    px1, py1, pz1, e1 = to_cartesian(pt1, eta1, phi1, mass1)
    px2, py2, pz2, e2 = to_cartesian(pt2, eta2, phi2, mass2)
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    e = e1 + e2

    return np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0))


def compute_best_z_candidate_masses(data, particle_type="mu", fill_value=-999.0):
    
    pt = data[f"{particle_type}_pt"]
    eta = data[f"{particle_type}_eta"]   #extracting particle kinematics
    phi = data[f"{particle_type}_phi"]

    #building momentum 4D objects
    particles = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": 0.0  #assuming massless for now
        },
        with_name="Momentum4D"
    )

    #need to check all unique lepton pairs
    pairs = ak.combinations(particles, 2, fields=["a", "b"])
    masses = (pairs["a"] + pairs["b"]).mass     #uses 4-vector addition and computes the momenta squared actually 

    #now we find the index of pair closest to Z mass (91.2 GeV)
    z_mass = 91.2
    mass_diff = abs(masses - z_mass)
    best_index = ak.argmin(mass_diff, axis=1)

    #must select the best candidate per event (or fill_value if no pairs exist)
    best_masses = ak.fill_none(
        ak.firsts(masses[ak.local_index(masses, axis=1) == best_index]),
        fill_value
    )

    return ak.to_numpy(best_masses)


def compute_best_z_candidate_pt(data, particle_type="mu", fill_value=-999.0):
    pt = data[f"{particle_type}_pt"]
    eta = data[f"{particle_type}_eta"]
    phi = data[f"{particle_type}_phi"]

    particles = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": 0.0
        },
        with_name="Momentum4D"
    )

    #all unique lepton pairs
    pairs = ak.combinations(particles, 2, fields=["a", "b"])
    z_cands = pairs["a"] + pairs["b"]   # full 4-vector of each Z candidate
    masses = z_cands.mass

    #pick the one closest to 91.2 GeV
    z_mass = 91.2
    mass_diff = abs(masses - z_mass)
    best_index = ak.argmin(mass_diff, axis=1)

    #extract the pt of the best candidate
    best_pt = ak.fill_none(
        ak.firsts(z_cands.pt[ak.local_index(z_cands.pt, axis=1) == best_index]),
        fill_value
    )

    return ak.to_numpy(best_pt)


def compute_delta_phi_Z_MET(data, particle_type="mu", fill_value=-999.0):
    pt = data[f"{particle_type}_pt"]
    eta = data[f"{particle_type}_eta"]
    phi = data[f"{particle_type}_phi"]

    particles = ak.zip(
        {"pt": pt, "eta": eta, "phi": phi, "mass": 0.0},
        with_name="Momentum4D"
    )

    #all unique lepton pairs
    pairs = ak.combinations(particles, 2, fields=["a", "b"])
    z_cands = pairs["a"] + pairs["b"]  # full 4-vector of each Z candidate
    masses = z_cands.mass

    z_mass = 91.2
    mass_diff = abs(masses - z_mass)
    best_index = ak.argmin(mass_diff, axis=1)

    #extract phi of the best candidate
    best_phi = ak.firsts(z_cands.phi[ak.local_index(z_cands.phi, axis=1) == best_index])

    #eplace missing phi with fill_value **before** delphi calculation
    best_phi = ak.fill_none(best_phi, fill_value)

    #delphi with MET
    phi_MET = data["MET_Phi"]
    dphi = np.arctan2(np.sin(best_phi - phi_MET), np.cos(best_phi - phi_MET))

    return ak.to_numpy(dphi)


def compute_delta_phi_Z_lep(data, particle_type="mu", fill_value=-999.0):
    import numpy as np
    import awkward as ak

    #first build leptons
    leptons = ak.zip(
        {
            "pt":  data[f"{particle_type}_pt"],
            "eta": data[f"{particle_type}_eta"],
            "phi": data[f"{particle_type}_phi"],
            "mass": 0.0,
        },
        with_name="Momentum4D",
    )

    #all dilepton pairs and Z-candidates
    pairs   = ak.combinations(leptons, 2, fields=["a", "b"])
    z_cands = pairs["a"] + pairs["b"]
    masses  = z_cands.mass

    #indexing of pair closest to mZ
    z_mass   = 91.2
    best_idx = ak.argmin(abs(masses - z_mass), axis=1)

    #grab index used in that best-Z pair
    lep_idx  = ak.local_index(leptons, axis=1)
    pair_idx = ak.combinations(lep_idx, 2, fields=["i", "j"])

    #collapsing best_i and best_j to scalars (per event)
    best_i = ak.firsts(pair_idx["i"][best_idx])
    best_j = ak.firsts(pair_idx["j"][best_idx])

    #masking the leptons not in the best-Z
    is_nonZ  = (lep_idx != best_i) & (lep_idx != best_j)
    nonZ_lep = ak.firsts(leptons[is_nonZ])  # leftover lepton (None if none)

    #best z four vector
    best_Z = ak.firsts(z_cands[best_idx])

    #delphi(best Z, leftover lepton), added in safe handling
    dphi = np.arctan2(
        np.sin(best_Z.phi - nonZ_lep.phi),
        np.cos(best_Z.phi - nonZ_lep.phi),
    )

    #filling in events where we don’t have a valid leftover lepton
    dphi = ak.fill_none(dphi, fill_value)

    return ak.to_numpy(dphi)






#helps plot delR dist, applies three different cuts, delR, pt and eta
def clean_leptons(events, deltaR_cut=0.4, pt_min=10, eta_max=2.5):
    electrons = ak.zip({
        "pt": events["el_pt"],
        "eta": events["el_eta"],
        "phi": events["el_phi"],
    }, with_name="PtEtaPhi")

    muons = ak.zip({
        "pt": events["mu_pt"],
        "eta": events["mu_eta"],
        "phi": events["mu_phi"],
    }, with_name="PtEtaPhi")

    jets = ak.zip({
        "eta": events["jet_eta"],
        "phi": events["jet_phi"],
    }, with_name="EtaPhi")

    #applying the kinematic cuts (preserving structure)
    ele_mask = (electrons.pt > pt_min) & (abs(electrons.eta) < eta_max)
    mu_mask = (muons.pt > pt_min) & (abs(muons.eta) < eta_max)

    electrons = ak.mask(electrons, ele_mask)
    muons = ak.mask(muons, mu_mask)




    # keep electrons far from all jets, cleaning it
    el_jet_pairs = ak.cartesian({"el": electrons, "jet": jets}, axis=1)
    deltaR_el_jet = delta_r(el_jet_pairs["el"].eta, el_jet_pairs["el"].phi,
                            el_jet_pairs["jet"].eta, el_jet_pairs["jet"].phi)
    keep_el = ak.all(deltaR_el_jet > deltaR_cut, axis=-1)
    cleaned_electrons = ak.mask(electrons, keep_el)

    #mu cleaning: keep muons far from all jets
    mu_jet_pairs = ak.cartesian({"mu": muons, "jet": jets}, axis=1)
    deltaR_mu_jet = delta_r(mu_jet_pairs["mu"].eta, mu_jet_pairs["mu"].phi,
                            mu_jet_pairs["jet"].eta, mu_jet_pairs["jet"].phi)
    keep_mu = ak.all(deltaR_mu_jet > deltaR_cut, axis=-1)
    cleaned_muons = ak.mask(muons, keep_mu)

    #keeping events with at least one lepton
    has_el = ak.num(ak.fill_none(cleaned_electrons, []), axis=1) > 0
    has_mu = ak.num(ak.fill_none(cleaned_muons, []), axis=1) > 0

    keep_events = has_el | has_mu

    #copying over filtered events and assigning cleaned leptons
    cleaned_events = copy.deepcopy(events[keep_events])
    cleaned_events["el_pt"] = cleaned_electrons[keep_events].pt
    cleaned_events["el_eta"] = cleaned_electrons[keep_events].eta
    cleaned_events["el_phi"] = cleaned_electrons[keep_events].phi
    cleaned_events["mu_pt"] = cleaned_muons[keep_events].pt
    cleaned_events["mu_eta"] = cleaned_muons[keep_events].eta
    cleaned_events["mu_phi"] = cleaned_muons[keep_events].phi

    return cleaned_events



def compute_invariant_mass_3lep(events):
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt)
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt)
        })
    ], axis=1)

    leptons = ak.Array(leptons, with_name="Momentum4D")
                                         #turning it into Lorentz vectors using vector's awkward extension

    leptons = ak.pad_none(leptons, 3)
    leptons3 = leptons[:, :3]
    m3lep = ak.sum(leptons3, axis=1).mass
    return ak.to_numpy(m3lep)

def compute_pairwise_masses_3lep(events):
    
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,   #building lepton collection (e + mu)
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt)
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt)
        }),
    ], axis=1)

    #need to tag the lorentz vectors
    leptons = ak.with_name(leptons, "Momentum4D")

    #we take first 3 leptons (and pad if fewer than 3)
    leptons = ak.pad_none(leptons, 3)[:, :3]

    #compute pairwise inv mass
    m12 = (leptons[:, 0] + leptons[:, 1]).mass
    m23 = (leptons[:, 1] + leptons[:, 2]).mass
    m31 = (leptons[:, 2] + leptons[:, 0]).mass

    return {
        "m12": ak.to_numpy(m12),
        "m23": ak.to_numpy(m23),
        "m31": ak.to_numpy(m31),
    }


def compute_delta_eta_phi_3lep(events):
    leptons = ak.concatenate([    #lep collection
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt)
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt)
        }),
    ], axis=1)

    #pad to 3 leptons
    leptons = ak.pad_none(leptons, 3)[:, :3]

    eta = leptons.eta
    phi = leptons.phi

    #deleta
    deta12 = eta[:,0] - eta[:,1]
    deta23 = eta[:,1] - eta[:,2]
    deta31 = eta[:,2] - eta[:,0]

    #del_phi(wrapped to [-pi, pi])
    dphi12 = np.arctan2(np.sin(phi[:,0] - phi[:,1]), np.cos(phi[:,0] - phi[:,1]))
    dphi23 = np.arctan2(np.sin(phi[:,1] - phi[:,2]), np.cos(phi[:,1] - phi[:,2]))
    dphi31 = np.arctan2(np.sin(phi[:,2] - phi[:,0]), np.cos(phi[:,2] - phi[:,0]))

    return {
        "deta12": ak.to_numpy(deta12),
        "deta23": ak.to_numpy(deta23),
        "deta31": ak.to_numpy(deta31),
        "dphi12": ak.to_numpy(dphi12),
        "dphi23": ak.to_numpy(dphi23),
        "dphi31": ak.to_numpy(dphi31),
    }


def compute_deltaR_3lep(events):
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt)
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt)
        }),
    ], axis=1)

    # pad to 3 leptons
    leptons = ak.pad_none(leptons, 3)[:, :3]

    eta = leptons.eta
    phi = leptons.phi

    # deleta
    deta12 = eta[:,0] - eta[:,1]
    deta23 = eta[:,1] - eta[:,2]
    deta31 = eta[:,2] - eta[:,0]

    #dphi (wrapped)
    dphi12 = np.arctan2(np.sin(phi[:,0] - phi[:,1]), np.cos(phi[:,0] - phi[:,1]))
    dphi23 = np.arctan2(np.sin(phi[:,1] - phi[:,2]), np.cos(phi[:,1] - phi[:,2]))
    dphi31 = np.arctan2(np.sin(phi[:,2] - phi[:,0]), np.cos(phi[:,2] - phi[:,0]))

    #deltaR
    deltaR12 = np.sqrt(deta12**2 + dphi12**2)
    deltaR23 = np.sqrt(deta23**2 + dphi23**2)
    deltaR31 = np.sqrt(deta31**2 + dphi31**2)

    return {
        "deltaR12": ak.to_numpy(deltaR12),
        "deltaR23": ak.to_numpy(deltaR23),
        "deltaR31": ak.to_numpy(deltaR31),
    }


    
def compute_event_shapes(events):
    leptons = ak.concatenate([
        ak.zip({"px": events.el_pt * np.cos(events.el_phi),
                "py": events.el_pt * np.sin(events.el_phi),
                "pz": events.el_pt * np.sinh(events.el_eta)}),
        ak.zip({"px": events.mu_pt * np.cos(events.mu_phi),
                "py": events.mu_pt * np.sin(events.mu_phi),
                "pz": events.mu_pt * np.sinh(events.mu_eta)})
    ], axis=1)

    #tensor for each event
    def shape_tensor(p):
        px, py, pz = p.px, p.py, p.pz
        p2 = px**2 + py**2 + pz**2
        s_xx = ak.sum(px * px / p2, axis=1)
        s_yy = ak.sum(py * py / p2, axis=1)
        s_zz = ak.sum(pz * pz / p2, axis=1)
        s_xy = ak.sum(px * py / p2, axis=1)
        s_xz = ak.sum(px * pz / p2, axis=1)
        s_yz = ak.sum(py * pz / p2, axis=1)
        return s_xx, s_yy, s_zz, s_xy, s_xz, s_yz

    s_xx, s_yy, s_zz, s_xy, s_xz, s_yz = shape_tensor(leptons)

                                        #for simplicity, we are returning the sum of diagonal terms, which is a proxy for sphericity
    sphericity = 1.5 * (s_yy + s_zz)
    aplanarity = 1.5 * s_zz  # crude approximation
    return ak.to_numpy(sphericity), ak.to_numpy(aplanarity)



def compute_event_shapes_exact(events):
    #first we build momentum arrays for all leptons (e + mu)
    leptons = ak.concatenate([
        ak.zip({
            "px": events.el_pt * np.cos(events.el_phi),
            "py": events.el_pt * np.sin(events.el_phi),
            "pz": events.el_pt * np.sinh(events.el_eta)
        }),
        ak.zip({
            "px": events.mu_pt * np.cos(events.mu_phi),
            "py": events.mu_pt * np.sin(events.mu_phi),
            "pz": events.mu_pt * np.sinh(events.mu_eta)
        }),
    ], axis=1)

    #computing |p|^2 for each lepton
    p2 = leptons.px**2 + leptons.py**2 + leptons.pz**2

    #summing of |p|^2 per event (denominator has been used for normalization)
    norm = ak.sum(p2, axis=1)

    #then compute the components of the momentum tensor numerator per event
    #sum_k p_i^(k) p_j^(k)
    s_xx = ak.sum(leptons.px * leptons.px, axis=1)
    s_yy = ak.sum(leptons.py * leptons.py, axis=1)
    s_zz = ak.sum(leptons.pz * leptons.pz, axis=1)
    s_xy = ak.sum(leptons.px * leptons.py, axis=1)
    s_xz = ak.sum(leptons.px * leptons.pz, axis=1)
    s_yz = ak.sum(leptons.py * leptons.pz, axis=1)

    #no of events
    n_events = len(events)

    #momentum tensor array [n_events, 3, 3] initialization
    S = np.zeros((n_events, 3, 3))

    #normalising and fill symmetric tensor
    S[:, 0, 0] = ak.to_numpy(s_xx / norm)
    S[:, 1, 1] = ak.to_numpy(s_yy / norm)
    S[:, 2, 2] = ak.to_numpy(s_zz / norm)
    S[:, 0, 1] = ak.to_numpy(s_xy / norm)
    S[:, 1, 0] = S[:, 0, 1]
    S[:, 0, 2] = ak.to_numpy(s_xz / norm)
    S[:, 2, 0] = S[:, 0, 2]
    S[:, 1, 2] = ak.to_numpy(s_yz / norm)
    S[:, 2, 1] = S[:, 1, 2]

    # eigenvalues for each event's tensor
    eigvals = np.linalg.eigvalsh(S)  # returns eigenvalues in ascending order

    #resorting, reversing actually so eigenvalues are descending lambda1 >= lambda2 >= lambda3
    eigvals = eigvals[:, ::-1]

    # sphericity and aplanarity computation
    sphericity = 1.5 * (eigvals[:, 1] + eigvals[:, 2])
    aplanarity = 1.5 * eigvals[:, 2]

    return sphericity, aplanarity

def transverse_mass(pt1, phi1, pt2, phi2):
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
    return np.sqrt(2 * pt1 * pt2 * (1 - np.cos(dphi)))

def total_event_et(events):        #total of lep, jet, MET, for each event, inclusive quantity, just simple sum
    #lep
    total_el_pt = ak.sum(events.el_pt, axis=1)  # electrons
    total_mu_pt = ak.sum(events.mu_pt, axis=1)  # muons

    #jets
    total_jet_pt = ak.sum(events.jet_pt, axis=1)

    #MET is a scalar per event
    total_met = events.MET

    #sum
    total_et = total_el_pt + total_mu_pt + total_jet_pt + total_met

    return ak.to_numpy(total_et, allow_missing=True)


def compute_mt(events):  #to find the transverse mass of the W candidate
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,          #first combining electrons and muons into one lepton collection
            "mass": 0.000511
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": 0.10566
        })
    ], axis=1)

    lep_pairs = ak.combinations(leptons, 2, fields=["lep1", "lep2"])  #lepton pairs (picks 2 without repetition)

    #inv mass
    def inv_mass(lep1, lep2):
        return np.sqrt(
            (lep1.mass + lep2.mass) ** 2 +
            2 * lep1.pt * lep2.pt *
            (np.cosh(lep1.eta - lep2.eta) - np.cos(lep1.phi - lep2.phi))
        )

    masses = inv_mass(lep_pairs["lep1"], lep_pairs["lep2"])

    #first find pair that is closest to the Z mass
    z_mass = 91.1876
    best_pair_idx = ak.argmin(np.abs(masses - z_mass), axis=1)
    best_pairs = lep_pairs[best_pair_idx]

    # mask those out that make up the best Z pair
    not_lep1 = ~((leptons.pt == best_pairs.lep1.pt) &
                 (leptons.eta == best_pairs.lep1.eta) &
                 (leptons.phi == best_pairs.lep1.phi))

    not_lep2 = ~((leptons.pt == best_pairs.lep2.pt) &
                 (leptons.eta == best_pairs.lep2.eta) &
                 (leptons.phi == best_pairs.lep2.phi))

    mask = not_lep1 & not_lep2
    final_lepton = leptons[mask]

    #ensuring that only 1 lepton per event comes out, the thing is this limits it. Need a more robust one where other leptons also should be checked.
    final_lepton = ak.firsts(final_lepton)

    #mT computation wit MET
    lep_pt = final_lepton.pt
    lep_phi = final_lepton.phi
    met_pt = events.MET
    met_phi = events.MET_Phi

    delta_phi = lep_phi - met_phi    #this actually reconstructs it with MET using the correct formula! Nice.
    mt = np.sqrt(2 * lep_pt * met_pt * (1 - np.cos(delta_phi)))

    return ak.to_numpy(mt)


def compute_pt_3lepsys(events):
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt),
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt),
        }),
    ], axis=1)

    #make them Lorentz vectors
    leptons = ak.Array(leptons, with_name="Momentum4D")

    #keep exactly 3
    leptons3 = ak.pad_none(leptons, 3)[:, :3]

    #sum 4-vectors, then get system pt
    lep3vec = ak.sum(leptons3, axis=1)
    return ak.to_numpy(lep3vec.pt)




def compute_dphi_met_3lepsys(events):
    
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,    #building leptons (electrons + muons)
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt),
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt),
        }),
    ], axis=1)

    #make lorentz vectors
    leptons = ak.Array(leptons, with_name="Momentum4D")

    #exactly 3 leptons (pad if fewer, cut if more)
    leptons3 = ak.pad_none(leptons, 3)[:, :3]

    #sum into a 3-lepton system
    lep3sys = ak.sum(leptons3, axis=1)

    #building MET vector
    met_vec = ak.zip({
        "pt": events.MET,
        "phi": events.MET_Phi,
        "eta": ak.zeros_like(events.MET),
        "mass": ak.zeros_like(events.MET),
    }, with_name="Momentum4D")

    #computation of del phi manually (vector doesn’t have delta_phi), this equation
    dphi = (lep3sys.phi - met_vec.phi + np.pi) % (2 * np.pi) - np.pi

    return ak.to_numpy(dphi)




def compute_thrust(events, n_steps=100):
    
    leptons = ak.concatenate([
        ak.zip({
            "px": events.el_pt * np.cos(events.el_phi),
            "py": events.el_pt * np.sin(events.el_phi),   #building array of 3-momenta for all leptons (e + mu)
            "pz": events.el_pt * np.sinh(events.el_eta)
        }),
        ak.zip({
            "px": events.mu_pt * np.cos(events.mu_phi),
            "py": events.mu_pt * np.sin(events.mu_phi),
            "pz": events.mu_pt * np.sinh(events.mu_eta)
        }),
    ], axis=1)

    #converting to numpy arrays (list of arrays for each event)
    leptons_np = ak.to_numpy(leptons)

    n_events = len(events)
    thrust_vals = np.zeros(n_events)

    #generating candidate directions (unit vectors) on sphere for each event
    # scan over theta and phi on a coarse grid to approximate max thrust
    thetas = np.linspace(0, np.pi, n_steps)
    phis = np.linspace(0, 2 * np.pi, n_steps)

    #precomputing sin, cos for speed
    sin_thetas = np.sin(thetas)
    cos_thetas = np.cos(thetas)
    sin_phis = np.sin(phis)
    cos_phis = np.cos(phis)

    for i in range(n_events):
        p = leptons_np[i]  # structured array

        if len(p) == 0:
            thrust_vals[i] = 0.5
            continue

    # convert structured array to normal 2D ndarray (n_leptons, 3)
        p_xyz = np.vstack([p['px'], p['py'], p['pz']]).T
        p_mag = np.linalg.norm(p_xyz, axis=1).sum()

        max_sum = 0
        for theta in thetas:
            for phi in phis:
                n_hat = np.array([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
            ])
                proj = np.abs(np.dot(p_xyz, n_hat)).sum()
                if proj > max_sum:
                    max_sum = proj
    
        thrust_vals[i] = max_sum / p_mag



def z_met_balance_features(data, particle_type="mu", fill_value=-999.0):
    pt  = data[f"{particle_type}_pt"]
    eta = data[f"{particle_type}_eta"]
    phi = data[f"{particle_type}_phi"]

    #massless lep 4-vectors (massless)
    leptons = ak.zip({"pt": pt, "eta": eta, "phi": phi, "mass": 0.0}, with_name="Momentum4D")

    #all unique pairs -> Z candidates
    pairs = ak.combinations(leptons, 2, fields=["a", "b"])
    z_cands = pairs["a"] + pairs["b"]
    masses  = z_cands.mass

    #picking th candidate closest to mZ
    z_mass = 91.2
    best_idx = ak.argmin(abs(masses - z_mass), axis=1)

    #besst Z candidate per event (may be None if no pairs)
    best_Z = ak.firsts(z_cands[ak.local_index(z_cands, axis=1) == best_idx])

    #MET 4-vector (eta=0, mass=0)
    metPT  = data["MET"]
    phiMET = data["MET_Phi"]

    #extract phi and pT for Z (keep None if missing)
    phi_Z  = ak.values_astype(best_Z.phi, float)
    pTZ    = ak.values_astype(best_Z.pt, float)

    #delphi in [-pi, pi]
    dphi = np.arctan2(np.sin(phi_Z - phiMET), np.cos(phi_Z - phiMET))

    #components of MET wrt Z
    met_par  = metPT * np.cos(dphi)
    met_perp = metPT * np.sin(dphi)

        #ratios/differences (guard against divide-by-zero)
    #ratios/differences (guard against divide-by-zero)
    safe_div = metPT / pTZ
    met_over_pTZ = ak.where(pTZ > 0, safe_div, np.nan)
    pTZ_minus_MET = pTZ - metPT



    #vector recoil | pT(Z) + MET |
    px_Z, py_Z = pTZ * np.cos(phi_Z), pTZ * np.sin(phi_Z)
    px_MET, py_MET = metPT * np.cos(phiMET), metPT * np.sin(phiMET)
    recoil = np.sqrt((px_Z + px_MET)**2 + (py_Z + py_MET)**2)

    #replace None with fill_value and convert to numpy
    out = {
        "dphi_Z_MET":     np.nan_to_num(ak.to_numpy(ak.fill_none(dphi, fill_value)), nan=fill_value, posinf=fill_value, neginf=fill_value),
        "met_par_to_Z":   np.nan_to_num(ak.to_numpy(ak.fill_none(met_par, fill_value)), nan=fill_value, posinf=fill_value, neginf=fill_value),
        "met_perp_to_Z":  np.nan_to_num(ak.to_numpy(ak.fill_none(met_perp, fill_value)), nan=fill_value, posinf=fill_value, neginf=fill_value),
        "met_over_pTZ":   np.nan_to_num(ak.to_numpy(ak.fill_none(met_over_pTZ, fill_value)), nan=fill_value, posinf=fill_value, neginf=fill_value),
        "pTZ_minus_MET":  np.nan_to_num(ak.to_numpy(ak.fill_none(pTZ_minus_MET, fill_value)), nan=fill_value, posinf=fill_value, neginf=fill_value),
        "recoil_Z_MET":   np.nan_to_num(ak.to_numpy(ak.fill_none(recoil, fill_value)), nan=fill_value, posinf=fill_value, neginf=fill_value),
    }
    return out



def mt_nonZ_lepton_MET(data, particle_type="mu", fill_value=-999.0):
    #first we collect leptons of given type
    pt  = data[f"{particle_type}_pt"]
    eta = data[f"{particle_type}_eta"]
    phi = data[f"{particle_type}_phi"]

    leptons = ak.zip(
        {"pt": pt, "eta": eta, "phi": phi, "mass": 0.0},
        with_name="Momentum4D"
    )

    #then build dilep pairs
    pairs = ak.combinations(leptons, 2, fields=["a", "b"])
    z_cands = pairs["a"] + pairs["b"]
    masses  = z_cands.mass

    #picking the the pair closest to mZ
    z_mass = 91.2
    best_idx = ak.argmin(abs(masses - z_mass), axis=1)


        #indices of leptons in the best-Z pair (per event)
    lep_idx = ak.local_index(leptons, axis=1)
    pair_idx = ak.combinations(lep_idx, 2, fields=["i", "j"])

    #extract i, j for the best-Z per event
    best_i = ak.singletons(pair_idx["i"][best_idx])
    best_j = ak.singletons(pair_idx["j"][best_idx])

    #masking for leptons not in the best-Z pair
    is_nonZ = (lep_idx != best_i) & (lep_idx != best_j)
    nonZ_lep = ak.firsts(leptons[is_nonZ])


    # MET
    met_pt  = data["MET"]
    met_phi = data["MET_Phi"]

    #requiring ≥3 leptons
    valid = ak.num(leptons) >= 3

    #computing transverse mass mT of leftover lepton with MET
    mt = ak.where(
        valid & ~ak.is_none(nonZ_lep),
        np.sqrt(
            2.0 * nonZ_lep.pt * met_pt *
            (1.0 - np.cos(nonZ_lep.phi - met_phi))
        ),
        None,
    )

    #as with others, return as flat numpy array with fill_value
    return ak.to_numpy(ak.fill_none(mt, fill_value))



###############################################################################################














#####################################################4lep#####################################
####################################################specific##################################
def extract_features_geq4lep(events, label, max_leptons=4, max_jets=2):

    def safe_get(arr, idx, default=-999):
        padded = ak.pad_none(arr, idx + 1, axis=1)
        extracted = padded[:, idx]
        return ak.to_numpy(ak.fill_none(extracted, default))

    def safe_get_jets(array, idx, max_items):
        padded = ak.pad_none(array, max_items, axis=1)
        extracted = padded[:, idx]
        return ak.to_numpy(extracted, allow_missing=True)

    lep_dvars = compute_delta_eta_phi_4lep(events)
    lep_pairmass = compute_pairwise_masses_4lep(events)
    lep_tripletmass = compute_triplet_masses_4lep(events)
    deltaR_4lep = compute_deltaR_4lep(events)
    #best_mass, actual_inv = compute_best_z_candidate_masses(events, particle_type="mu", z_window=10.0)
    sph, apl = compute_event_shapes(events)
    features = {
        #leading leptons (electron and muon per index)
        "el_pt_0": safe_get(events.el_pt, 0, -999),
        "el_eta_0": safe_get(events.el_eta, 0, -999),
        "el_phi_0": safe_get(events.el_phi, 0, -999),

        "mu_pt_0": safe_get(events.mu_pt, 0, -999),
        "mu_eta_0": safe_get(events.mu_eta, 0, -999),
        "mu_phi_0": safe_get(events.mu_phi, 0, -999),

        "el_pt_1": safe_get(events.el_pt, 1, -999),
        "el_eta_1": safe_get(events.el_eta, 1, -999),
        "el_phi_1": safe_get(events.el_phi, 1, -999),

        "mu_pt_1": safe_get(events.mu_pt, 1, -999),
        "mu_eta_1": safe_get(events.mu_eta, 1, -999),
        "mu_phi_1": safe_get(events.mu_phi, 1, -999),

        "el_pt_2": safe_get(events.el_pt, 2, -999),
        "el_eta_2": safe_get(events.el_eta, 2, -999),
        "el_phi_2": safe_get(events.el_phi, 2, -999),

        "mu_pt_2": safe_get(events.mu_pt, 2, -999),
        "mu_eta_2": safe_get(events.mu_eta, 2, -999),
        "mu_phi_2": safe_get(events.mu_phi, 2, -999),

        #4th lepton
        "el_pt_3": safe_get(events.el_pt, 3, -999),
        "el_eta_3": safe_get(events.el_eta, 3, -999),
        "el_phi_3": safe_get(events.el_phi, 3, -999),

        "mu_pt_3": safe_get(events.mu_pt, 3, -999),
        "mu_eta_3": safe_get(events.mu_eta, 3, -999),
        "mu_phi_3": safe_get(events.mu_phi, 3, -999),

        #jets
        "jet_pt_0": safe_get_jets(events.jet_pt, 0, max_jets),
        "jet_eta_0": safe_get_jets(events.jet_eta, 0, max_jets),
        "jet_phi_0": safe_get_jets(events.jet_phi, 0, max_jets),

        "jet_pt_1": safe_get_jets(events.jet_pt, 1, max_jets),
        "jet_eta_1": safe_get_jets(events.jet_eta, 1, max_jets),
        "jet_phi_1": safe_get_jets(events.jet_phi, 1, max_jets),
        "jet_mass_0": safe_get_jets(events.jet_mass, 0, max_jets),
        "jet_btag_0": safe_get_jets(events.jet_btag, 0, max_jets),

        #other features
        "MET": ak.to_numpy(events.MET, allow_missing=True),
        "MET_Phi": ak.to_numpy(events.MET_Phi, allow_missing=True),

        # 4-lepton invariant mass (useful for ZZ/H->ZZ)
        "m_4lep": compute_invariant_mass_4lep(events),
        
        "total_event_et": total_event_et_ge4lep(events),

        "MT_lep_MET" : compute_mt_4l(events),

        "pt_4lepsys": compute_pt_4lepsys(events),
        "dphi_met_4lepsys": compute_dphi_met_4lepsys(events),

        
        #best Z candidates (pairing leptons closest to Z mass)
        "best_Z1_mass_mu": compute_best_z_candidate_masses_4lep(events, particle_type="mu", pair_index=0),
        "best_Z2_mass_mu": compute_best_z_candidate_masses_4lep(events, particle_type="mu", pair_index=1),
        "best_Z1_mass_el": compute_best_z_candidate_masses_4lep(events, particle_type="el", pair_index=0),
        "best_Z2_mass_el": compute_best_z_candidate_masses_4lep(events, particle_type="el", pair_index=1),

        "del_phi_bestZs": compute_delta_phi_between_Zs(events),

        #dilepton invariant masses (all unique pairs)
        "m12": lep_pairmass["m12"],
        "m13": lep_pairmass["m13"],
        "m14": lep_pairmass["m14"],
        "m23": lep_pairmass["m23"],
        "m24": lep_pairmass["m24"],
        "m34": lep_pairmass["m34"],

        #trilepton invariant masses (useful for WZ-like topologies)
        "m123": lep_tripletmass["m123"],
        "m124": lep_tripletmass["m124"],
        "m134": lep_tripletmass["m134"],
        "m234": lep_tripletmass["m234"],

        #del_eta, del_phi between lepton pairs
        "deta12": lep_dvars["deta12"],
        "deta13": lep_dvars["deta13"],
        "deta14": lep_dvars["deta14"],
        "deta23": lep_dvars["deta23"],
        "deta24": lep_dvars["deta24"],
        "deta34": lep_dvars["deta34"],

        "dphi12": lep_dvars["dphi12"],
        "dphi13": lep_dvars["dphi13"],
        "dphi14": lep_dvars["dphi14"],
        "dphi23": lep_dvars["dphi23"],
        "dphi24": lep_dvars["dphi24"],
        "dphi34": lep_dvars["dphi34"],

        
        "deltaR12": deltaR_4lep["deltaR12"],
        "deltaR13": deltaR_4lep["deltaR13"],
        "deltaR14": deltaR_4lep["deltaR14"],
        "deltaR23": deltaR_4lep["deltaR23"],
        "deltaR24": deltaR_4lep["deltaR24"],
        "deltaR34": deltaR_4lep["deltaR34"],

            #event-shape variables
        "sphericity": sph,
        "aplanarity": apl,

        # Transverse mass with MET (for leptons + MET signatures)
        #"MT_lep_MET": compute_mt(events)    #not sure what the remaining lepton would be here...this variable might not besuper relevant
}

        #printing  some debug outputs for inspection
    print("m_4lep:", features["m_4lep"][:5])
    print("total_event_et:", features["total_event_et"][:5])
    print("MT_lep_MET:", features["MT_lep_MET"][:5])

    print("best_Z1_mass_mu:", features["best_Z1_mass_mu"][:5])
    print("best_Z2_mass_mu:", features["best_Z2_mass_mu"][:5])
    print("best_Z1_mass_el:", features["best_Z1_mass_el"][:5])
    print("best_Z2_mass_el:", features["best_Z2_mass_el"][:5])
    

    print("m12:", ak.to_numpy(features["m12"], allow_missing=True)[:5])
    print("m13:", ak.to_numpy(features["m34"], allow_missing=True)[:5])
    print("m14:", ak.to_numpy(features["m12"], allow_missing=True)[:5])
    print("m23:", ak.to_numpy(features["m34"], allow_missing=True)[:5])
    print("m24:", ak.to_numpy(features["m12"], allow_missing=True)[:5])
    print("m34:", ak.to_numpy(features["m34"], allow_missing=True)[:5])
    
    print("m123:", ak.to_numpy(features["m123"], allow_missing=True)[:5])
    print("m124:", ak.to_numpy(features["m123"], allow_missing=True)[:5])
    print("m134:", ak.to_numpy(features["m123"], allow_missing=True)[:5])
    print("m234:", ak.to_numpy(features["m234"], allow_missing=True)[:5])
        
    print("deta12:", ak.to_numpy(features["deta12"], allow_missing=True)[:5])
    print("deta13:", ak.to_numpy(features["deta34"], allow_missing=True)[:5])
    print("deta14:", ak.to_numpy(features["dphi12"], allow_missing=True)[:5])
    print("deta23:", ak.to_numpy(features["dphi34"], allow_missing=True)[:5])    
    print("deta24:", ak.to_numpy(features["deta12"], allow_missing=True)[:5])
    print("deta34:", ak.to_numpy(features["deta34"], allow_missing=True)[:5])
    print("dphi12:", ak.to_numpy(features["dphi12"], allow_missing=True)[:5])
    print("dphi13:", ak.to_numpy(features["dphi34"], allow_missing=True)[:5])  
    print("dphi14:", ak.to_numpy(features["deta12"], allow_missing=True)[:5])
    print("dphi23:", ak.to_numpy(features["deta34"], allow_missing=True)[:5])
    print("dphi24:", ak.to_numpy(features["dphi12"], allow_missing=True)[:5])
    print("dphi34:", ak.to_numpy(features["dphi34"], allow_missing=True)[:5])  
    
    print("sphericity:", features["sphericity"][:5])
    print("aplanarity:", features["aplanarity"][:5])

    lengths = [len(arr) for arr in features.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent array lengths in the features: {set(lengths)}")

    df = pd.DataFrame(features)
    df["label"] = label
    return df


vector.register_awkward()

def compute_invariant_mass_4lep(events):
    leptons = ak.concatenate([   #build lep 4 vectors with el, mu
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt),
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt),
        }),
    ], axis=1)

    #turning them into lorentz vectors using vector’s awkward extension
    leptons = ak.Array(leptons, with_name="Momentum4D")

    #this ensures at least 4 leptons per event, take the first 4
    leptons = ak.pad_none(leptons, 4)
    leptons4 = leptons[:, :4]

    #invariant mass of the 4-lepton system
    m4lep = ak.sum(leptons4, axis=1).mass

    return ak.to_numpy(m4lep)



#computing invariant masses of the best Z candidates formed from 4 leptons
def compute_best_z_candidate_masses_4lep(data, particle_type="mu", pair_index=0, fill_value=-999.0):
    
    pt = data[f"{particle_type}_pt"]
    eta = data[f"{particle_type}_eta"]  #extract particle kinematics
    phi = data[f"{particle_type}_phi"]

    #momentum4D objects
    leptons = ak.zip(
        {
            "pt": pt,
            "eta": eta,
            "phi": phi,
            "mass": 0.0  # massless leptons
        },
        with_name="Momentum4D"
    )

    #ensuring we have at least 4 leptons
    leptons = ak.pad_none(leptons, 4)
    leptons4 = leptons[:, :4]

    #ALL UNIQUE lepton pairs
    pairs = ak.combinations(leptons4, 2, fields=["a", "b"])
    masses = (pairs["a"] + pairs["b"]).mass  # invariant masses

    #best Z candidate
    z_mass = 91.2
    mass_diff = abs(masses - z_mass)
    sorted_indices = ak.argsort(mass_diff, axis=1)

    # pair_index-th best Z candidate 
    chosen_index = sorted_indices[:, pair_index]

    #get mass
    chosen_mass = ak.fill_none(
        masses[ak.local_index(masses, axis=1) == chosen_index],
        fill_value
    )

    return ak.to_numpy(ak.firsts(chosen_mass))




def compute_pairwise_masses_4lep(events):
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,    #building lepton collection (electrons + muons)
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt)
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt)
        }),
    ], axis=1)

    #Tag as l vectors
    leptons = ak.with_name(leptons, "Momentum4D")

    #padding if fewer than 4 leptons as before
    leptons = ak.pad_none(leptons, 4)[:, :4]

    #six unique pairwise m
    m12 = (leptons[:, 0] + leptons[:, 1]).mass
    m13 = (leptons[:, 0] + leptons[:, 2]).mass
    m14 = (leptons[:, 0] + leptons[:, 3]).mass
    m23 = (leptons[:, 1] + leptons[:, 2]).mass
    m24 = (leptons[:, 1] + leptons[:, 3]).mass
    m34 = (leptons[:, 2] + leptons[:, 3]).mass

    return {
        "m12": ak.to_numpy(m12),
        "m13": ak.to_numpy(m13),
        "m14": ak.to_numpy(m14),
        "m23": ak.to_numpy(m23),
        "m24": ak.to_numpy(m24),
        "m34": ak.to_numpy(m34),
    }




def compute_triplet_masses_4lep(events):
   
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,     #make lepton collection (electrons + muons)
            "mass": ak.zeros_like(events.el_pt)
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt)
        }),
    ], axis=1)

    #tag as L vectors
    leptons = ak.with_name(leptons, "Momentum4D")

    #pad if fewer than 4 
    leptons = ak.pad_none(leptons, 4)[:, :4]

    #four unique ones
    m123 = (leptons[:, 0] + leptons[:, 1] + leptons[:, 2]).mass
    m124 = (leptons[:, 0] + leptons[:, 1] + leptons[:, 3]).mass
    m134 = (leptons[:, 0] + leptons[:, 2] + leptons[:, 3]).mass
    m234 = (leptons[:, 1] + leptons[:, 2] + leptons[:, 3]).mass

    return {
        "m123": ak.to_numpy(m123),
        "m124": ak.to_numpy(m124),
        "m134": ak.to_numpy(m134),
        "m234": ak.to_numpy(m234),
    }


def total_event_et_ge4lep(events):
    #counting total leptons per event
    #n_lep = ak.num(events.el_pt, axis=1) + ak.num(events.mu_pt, axis=1)

    #apply mask: only keep events with ≥ 4 leptons
    #mask = n_lep >= 4
    #events_sel = events[mask]

    #els
    total_el_pt = ak.sum(events.el_pt, axis=1)
    #mus
    total_mu_pt = ak.sum(events.mu_pt, axis=1)
    #jets
    total_jet_pt = ak.sum(events.jet_pt, axis=1)
    #met
    total_met = events.MET

    #s
    total_et = total_el_pt + total_mu_pt + total_jet_pt + total_met

    return ak.to_numpy(total_et, allow_missing=True)





def compute_mt_4l(events):
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": 0.000511
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": 0.10566
        })
    ], axis=1)

    #sll lep pairs
    lep_pairs = ak.combinations(leptons, 2, fields=["lep1", "lep2"])

    #inv mass
    def inv_mass(lep1, lep2):
        return np.sqrt(
            (lep1.mass + lep2.mass) ** 2 +
            2 * lep1.pt * lep2.pt *
            (np.cosh(lep1.eta - lep2.eta) - np.cos(lep1.phi - lep2.phi))
        )

    masses = inv_mass(lep_pairs["lep1"], lep_pairs["lep2"])

    #best z
    z_mass = 91.1876
    best_pair_idx = ak.argmin(np.abs(masses - z_mass), axis=1)
    best_pairs = lep_pairs[best_pair_idx]

    #wrapping best-pair leptons to align shapes with `leptons`
    lep1 = ak.firsts(best_pairs.lep1)
    lep2 = ak.firsts(best_pairs.lep2)


    #mask to exclude Z leptons
    mask = ~(
    ((leptons.pt == lep1.pt) &
     (leptons.eta == lep1.eta) &
     (leptons.phi == lep1.phi)) |
    ((leptons.pt == lep2.pt) &
     (leptons.eta == lep2.eta) &
     (leptons.phi == lep2.phi)))


    leftover_leptons = leptons[mask]

    #then we pick the highest-pT leftover lepton
    leftover_leptons = ak.firsts(leftover_leptons[ak.argsort(leftover_leptons.pt, ascending=False)])

    #mT with MET
    lep_pt = leftover_leptons.pt
    lep_phi = leftover_leptons.phi
    met_pt = events.MET
    met_phi = events.MET_Phi

    delta_phi = lep_phi - met_phi
    mt = np.sqrt(2 * lep_pt * met_pt * (1 - np.cos(delta_phi)))

    return ak.to_numpy(mt)




    
def compute_delta_eta_phi_4lep(events):
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt),
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt),
        }),
    ], axis=1)

    #we pad to 4 lep
    leptons = ak.pad_none(leptons, 4)[:, :4]

    #extract
    eta = leptons.eta
    phi = leptons.phi

    #del_eta
    deta12 = eta[:, 0] - eta[:, 1]
    deta13 = eta[:, 0] - eta[:, 2]
    deta14 = eta[:, 0] - eta[:, 3]
    deta23 = eta[:, 1] - eta[:, 2]
    deta24 = eta[:, 1] - eta[:, 3]
    deta34 = eta[:, 2] - eta[:, 3]

    #del_phi (wrapped into [-pi,pi])
    def delta_phi(phi1, phi2):
        return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))

    dphi12 = delta_phi(phi[:, 0], phi[:, 1])
    dphi13 = delta_phi(phi[:, 0], phi[:, 2])
    dphi14 = delta_phi(phi[:, 0], phi[:, 3])
    dphi23 = delta_phi(phi[:, 1], phi[:, 2])
    dphi24 = delta_phi(phi[:, 1], phi[:, 3])
    dphi34 = delta_phi(phi[:, 2], phi[:, 3])

    return {
        "deta12": ak.to_numpy(deta12),
        "deta13": ak.to_numpy(deta13),
        "deta14": ak.to_numpy(deta14),
        "deta23": ak.to_numpy(deta23),
        "deta24": ak.to_numpy(deta24),
        "deta34": ak.to_numpy(deta34),
        "dphi12": ak.to_numpy(dphi12),
        "dphi13": ak.to_numpy(dphi13),
        "dphi14": ak.to_numpy(dphi14),
        "dphi23": ak.to_numpy(dphi23),
        "dphi24": ak.to_numpy(dphi24),
        "dphi34": ak.to_numpy(dphi34),
    }


def compute_pt_4lepsys(events):
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt),
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt),
        }),
    ], axis=1)

    #make Lorentz vectors
    leptons = ak.Array(leptons, with_name="Momentum4D")

    #keep exactly 4 leptons (pad if fewer, cut if more)
    leptons4 = ak.pad_none(leptons, 4)[:, :4]

    # sum 4-vectors, then get system pt
    lep4vec = ak.sum(leptons4, axis=1)
    return ak.to_numpy(lep4vec.pt)



def compute_dphi_met_4lepsys(events):
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt),
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt),
        }),
    ], axis=1)

    #make Lorentz vectors
    leptons = ak.Array(leptons, with_name="Momentum4D")

    #exactly 4 leptons (pad if fewer, cut if more), just to make sure
    leptons4 = ak.pad_none(leptons, 4)[:, :4]

    #sum into a 4-lepton system
    lep4sys = ak.sum(leptons4, axis=1)

    # build MET vector
    met_vec = ak.zip({
        "pt": events.MET,
        "phi": events.MET_Phi,
        "eta": ak.zeros_like(events.MET),
        "mass": ak.zeros_like(events.MET),
    }, with_name="Momentum4D")

    # delphi between system and MET
    dphi = (lep4sys.phi - met_vec.phi + np.pi) % (2 * np.pi) - np.pi

    return ak.to_numpy(dphi)




def compute_deltaR_4lep(events):
    #built leptons (electrons + muons)
    leptons = ak.concatenate([
        ak.zip({
            "pt": events.el_pt,
            "eta": events.el_eta,
            "phi": events.el_phi,
            "mass": ak.zeros_like(events.el_pt)
        }),
        ak.zip({
            "pt": events.mu_pt,
            "eta": events.mu_eta,
            "phi": events.mu_phi,
            "mass": ak.zeros_like(events.mu_pt)
        }),
    ], axis=1)

    #pad to at least 4 leptons, truncate extras
    leptons = ak.pad_none(leptons, 4)[:, :4]

    eta = leptons.eta
    phi = leptons.phi

    #comp all 6 delR combinations
    deltaR = {}
    pairs = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    for i, j in pairs:
        deta = eta[:, i] - eta[:, j]
        dphi = np.arctan2(np.sin(phi[:, i] - phi[:, j]), np.cos(phi[:, i] - phi[:, j]))
        deltaR[f"deltaR{i+1}{j+1}"] = ak.to_numpy(np.sqrt(deta**2 + dphi**2))

    return deltaR



def compute_delta_phi_between_Zs(data, fill_value=-999.0):
    
    #Compute delphi between the two best Z candidates from up to 4 leptons
    #(electrons + muons), ensuring the Zs do not share leptons.
    # Electrons
    electrons = ak.zip(
        {
            "pt": data["el_pt"],
            "eta": data["el_eta"],
            "phi": data["el_phi"],
            "mass": 0.000511,
        },
        with_name="Momentum4D",
    )

    # Muons
    muons = ak.zip(
        {
            "pt": data["mu_pt"],
            "eta": data["mu_eta"],
            "phi": data["mu_phi"],
            "mass": 0.105,
        },
        with_name="Momentum4D",
    )

    #merge leptons
    leptons = ak.concatenate([electrons, muons], axis=1)

    #reeequire at least 4 leptons, take leading 4
    leptons = ak.pad_none(leptons, 4)

    #rebuild as Momentum4D (to preserve vector behavior after slicing)
    leptons4 = ak.zip(
        {
            "pt": leptons.pt[:, :4],
            "eta": leptons.eta[:, :4],
            "phi": leptons.phi[:, :4],
            "mass": leptons.mass[:, :4],
        },
        with_name="Momentum4D",
    )

    # unique disjoint pairings
    pairings = [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]

    z_mass = 91.2
    best_pairing = None
    best_score = None

    for p1, p2 in pairings:
        z1 = leptons4[:, p1[0]] + leptons4[:, p1[1]]
        z2 = leptons4[:, p2[0]] + leptons4[:, p2[1]]

        m1 = z1.mass
        m2 = z2.mass

        #score: closeness of both masses to mZ
        score = abs(m1 - z_mass) + abs(m2 - z_mass)

        if best_score is None:
            best_score = score
            best_pairing = (z1, z2)
        else:
            mask = score < best_score
            best_score = ak.where(mask, score, best_score)
            best_pairing = (
                ak.where(mask, z1, best_pairing[0]),
                ak.where(mask, z2, best_pairing[1]),
            )

    #delphi between the two Z candidates
    dphi = best_pairing[0].deltaphi(best_pairing[1])
    dphi = ak.fill_none(dphi, fill_value)

    return ak.to_numpy(dphi)


#######################################################################################################
###############################5lep####################################################################

def extract_features_5lep(events, label, max_leptons=5, max_jets=2):

    def safe_get(arr, idx, default=-999):
        padded = ak.pad_none(arr, idx + 1, axis=1)
        extracted = padded[:, idx]
        return ak.to_numpy(ak.fill_none(extracted, default))

    def safe_get_jets(array, idx, max_items):
        padded = ak.pad_none(array, max_items, axis=1)
        extracted = padded[:, idx]
        return ak.to_numpy(extracted, allow_missing=True)

    lep_dvars = compute_delta_eta_phi_4lep(events)
    lep_pairmass = compute_pairwise_masses_4lep(events)
    lep_tripletmass = compute_triplet_masses_4lep(events)
    deltaR_4lep = compute_deltaR_4lep(events)
    #best_mass, actual_inv = compute_best_z_candidate_masses(events, particle_type="mu", z_window=10.0)
    sph, apl = compute_event_shapes(events)
    features = {
        #leading leptons (electron and muon per index)
        "el_pt_0": safe_get(events.el_pt, 0, -999),
        "el_eta_0": safe_get(events.el_eta, 0, -999),
        "el_phi_0": safe_get(events.el_phi, 0, -999),

        "mu_pt_0": safe_get(events.mu_pt, 0, -999),
        "mu_eta_0": safe_get(events.mu_eta, 0, -999),
        "mu_phi_0": safe_get(events.mu_phi, 0, -999),

        "el_pt_1": safe_get(events.el_pt, 1, -999),
        "el_eta_1": safe_get(events.el_eta, 1, -999),
        "el_phi_1": safe_get(events.el_phi, 1, -999),

        "mu_pt_1": safe_get(events.mu_pt, 1, -999),
        "mu_eta_1": safe_get(events.mu_eta, 1, -999),
        "mu_phi_1": safe_get(events.mu_phi, 1, -999),

        "el_pt_2": safe_get(events.el_pt, 2, -999),
        "el_eta_2": safe_get(events.el_eta, 2, -999),
        "el_phi_2": safe_get(events.el_phi, 2, -999),

        "mu_pt_2": safe_get(events.mu_pt, 2, -999),
        "mu_eta_2": safe_get(events.mu_eta, 2, -999),
        "mu_phi_2": safe_get(events.mu_phi, 2, -999),

        #4th lepton
        "el_pt_3": safe_get(events.el_pt, 3, -999),
        "el_eta_3": safe_get(events.el_eta, 3, -999),
        "el_phi_3": safe_get(events.el_phi, 3, -999),

        "mu_pt_3": safe_get(events.mu_pt, 3, -999),
        "mu_eta_3": safe_get(events.mu_eta, 3, -999),
        "mu_phi_3": safe_get(events.mu_phi, 3, -999),

        #5th lepton
        "el_pt_4": safe_get(events.el_pt, 3, -999),
        "el_eta_4": safe_get(events.el_eta, 3, -999),
        "el_phi_4": safe_get(events.el_phi, 3, -999),

        "mu_pt_4": safe_get(events.mu_pt, 3, -999),
        "mu_eta_4": safe_get(events.mu_eta, 3, -999),
        "mu_phi_4": safe_get(events.mu_phi, 3, -999),

        #jets
        "jet_pt_0": safe_get_jets(events.jet_pt, 0, max_jets),
        "jet_eta_0": safe_get_jets(events.jet_eta, 0, max_jets),
        "jet_phi_0": safe_get_jets(events.jet_phi, 0, max_jets),

        "jet_pt_1": safe_get_jets(events.jet_pt, 1, max_jets),
        "jet_eta_1": safe_get_jets(events.jet_eta, 1, max_jets),
        "jet_phi_1": safe_get_jets(events.jet_phi, 1, max_jets),
        "jet_mass_0": safe_get_jets(events.jet_mass, 0, max_jets),
        "jet_btag_0": safe_get_jets(events.jet_btag, 0, max_jets),

        #other features
        "MET": ak.to_numpy(events.MET, allow_missing=True),
        "MET_Phi": ak.to_numpy(events.MET_Phi, allow_missing=True),

        # 5-lepton invariant mass (useful for ZZ/H->ZZ)
        #"m_5lep": compute_invariant_mass_4lep(events),
        
        #"total_event_et": total_event_et_ge4lep(events),

        #"MT_lep_MET" : compute_mt_4l(events),

        #"pt_5lepsys": compute_pt_4lepsys(events),
        #"dphi_met_5lepsys": compute_dphi_met_4lepsys(events),

        
        #best Z candidates (pairing leptons closest to Z mass)
        #"best_Z1_mass_mu": compute_best_z_candidate_masses_4lep(events, particle_type="mu", pair_index=0),
        #"best_Z2_mass_mu": compute_best_z_candidate_masses_4lep(events, particle_type="mu", pair_index=1),
        #"best_Z1_mass_el": compute_best_z_candidate_masses_4lep(events, particle_type="el", pair_index=0),
        #"best_Z2_mass_el": compute_best_z_candidate_masses_4lep(events, particle_type="el", pair_index=1),

        #"del_phi_bestZs": compute_delta_phi_between_Zs(events),

        #dilepton invariant masses (all unique pairs)
        #"m12": lep_pairmass["m12"],
        #"m13": lep_pairmass["m13"],
        #"m14": lep_pairmass["m14"],
        #"m23": lep_pairmass["m23"],
        #"m24": lep_pairmass["m24"],
        #"m34": lep_pairmass["m34"],

        #trilepton invariant masses (useful for WZ-like topologies)
        #"m123": lep_tripletmass["m123"],
        #"m124": lep_tripletmass["m124"],
        #"m134": lep_tripletmass["m134"],
        #"m234": lep_tripletmass["m234"],

        #del_eta, del_phi between lepton pairs
        #"deta12": lep_dvars["deta12"],
        #"deta13": lep_dvars["deta13"],
        #"deta14": lep_dvars["deta14"],
        #"deta23": lep_dvars["deta23"],
        #"deta24": lep_dvars["deta24"],
        #"deta34": lep_dvars["deta34"],

        #"dphi12": lep_dvars["dphi12"],
        #"dphi13": lep_dvars["dphi13"],
        #"dphi14": lep_dvars["dphi14"],
        #"dphi23": lep_dvars["dphi23"],
        #"dphi24": lep_dvars["dphi24"],
        #"dphi34": lep_dvars["dphi34"],

        
       # "deltaR12": deltaR_4lep["deltaR12"],
        #"deltaR13": deltaR_4lep["deltaR13"],
        #"deltaR14": deltaR_4lep["deltaR14"],
        #"deltaR23": deltaR_4lep["deltaR23"],
        #"deltaR24": deltaR_4lep["deltaR24"],
        #"deltaR34": deltaR_4lep["deltaR34"],

            #event-shape variables
        #"sphericity": sph,
        #"aplanarity": apl,

        # Transverse mass with MET (for leptons + MET signatures)
        #"MT_lep_MET": compute_mt(events)    #not sure what the remaining lepton would be here...this variable might not besuper relevant
}

        #printing  some debug outputs for inspection
   # print("m_5lep:", features["m_4lep"][:5])
   # print("total_event_et:", features["total_event_et"][:5])
    #print("MT_lep_MET:", features["MT_lep_MET"][:5])

    #print("best_Z1_mass_mu:", features["best_Z1_mass_mu"][:5])
    #print("best_Z2_mass_mu:", features["best_Z2_mass_mu"][:5])
    #print("best_Z1_mass_el:", features["best_Z1_mass_el"][:5])
    #print("best_Z2_mass_el:", features["best_Z2_mass_el"][:5])
    

    #print("m12:", ak.to_numpy(features["m12"], allow_missing=True)[:5])
    #print("m13:", ak.to_numpy(features["m34"], allow_missing=True)[:5])
    #print("m14:", ak.to_numpy(features["m12"], allow_missing=True)[:5])
    #print("m23:", ak.to_numpy(features["m34"], allow_missing=True)[:5])
    #print("m24:", ak.to_numpy(features["m12"], allow_missing=True)[:5])
    #print("m34:", ak.to_numpy(features["m34"], allow_missing=True)[:5])
    
    #print("m123:", ak.to_numpy(features["m123"], allow_missing=True)[:5])
    #print("m124:", ak.to_numpy(features["m123"], allow_missing=True)[:5])
    #print("m134:", ak.to_numpy(features["m123"], allow_missing=True)[:5])
    #print("m234:", ak.to_numpy(features["m234"], allow_missing=True)[:5])
        
    #print("deta12:", ak.to_numpy(features["deta12"], allow_missing=True)[:5])
    #print("deta13:", ak.to_numpy(features["deta34"], allow_missing=True)[:5])
    #print("deta14:", ak.to_numpy(features["dphi12"], allow_missing=True)[:5])
    #print("deta23:", ak.to_numpy(features["dphi34"], allow_missing=True)[:5])    
    #print("deta24:", ak.to_numpy(features["deta12"], allow_missing=True)[:5])
    #print("deta34:", ak.to_numpy(features["deta34"], allow_missing=True)[:5])
    #print("dphi12:", ak.to_numpy(features["dphi12"], allow_missing=True)[:5])
    #print("dphi13:", ak.to_numpy(features["dphi34"], allow_missing=True)[:5])  
    #print("dphi14:", ak.to_numpy(features["deta12"], allow_missing=True)[:5])
    #print("dphi23:", ak.to_numpy(features["deta34"], allow_missing=True)[:5])
    #print("dphi24:", ak.to_numpy(features["dphi12"], allow_missing=True)[:5])
    #print("dphi34:", ak.to_numpy(features["dphi34"], allow_missing=True)[:5])  
    
    #print("sphericity:", features["sphericity"][:5])
    #print("aplanarity:", features["aplanarity"][:5])

    lengths = [len(arr) for arr in features.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"Inconsistent array lengths in the features: {set(lengths)}")

    df = pd.DataFrame(features)
    df["label"] = label
    return df


vector.register_awkward()
