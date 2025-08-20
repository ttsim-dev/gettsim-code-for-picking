"""
Shared configuration and utilities for GETTSIM/TTSIM benchmarking.

This module contains all the shared constants, configurations, and utility functions
used by both benchmark.py and benchmark_profile.py to eliminate code duplication.
"""

import gc
import os
import time
import threading
import psutil
from datetime import datetime

# Import GETTSIM/TTSIM components
from gettsim import InputData, MainTarget, TTTargets, Labels, SpecializedEnvironment, RawResults
from gettsim import germany
import ttsim
from ttsim.main_args import OrigPolicyObjects

# JAX-specific imports for cache management
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


# =============================================================================
# GETTSIM MAIN WRAPPER
# =============================================================================

def main(**kwargs):
    """Wrapper around ttsim.main that automatically sets the German root path and supports tt_function."""
    # Set German tax system as default if no orig_policy_objects provided
    if kwargs.get('orig_policy_objects') is None:
        kwargs['orig_policy_objects'] = OrigPolicyObjects(root=germany.ROOT_PATH)
    
    return ttsim.main(**kwargs)


# =============================================================================
# TAX-TRANSFER TARGETS CONFIGURATION
# =============================================================================

TT_TARGETS = {
    "einkommensteuer": {
        "betrag_m_sn": "income_tax_m",
        "zu_versteuerndes_einkommen_y_sn": "taxable_income_y_sn",
    },
    "sozialversicherung": {
        "pflege": {
            "beitrag": {
                "betrag_versicherter_m": "long_term_care_insurance_contribution_m",
            },
        },
        "kranken": {
            "beitrag": {"betrag_versicherter_m": "health_insurance_contribution_m"},
        },
        "rente": {
            "beitrag": {"betrag_versicherter_m": "pension_insurance_contribution_m"},
            "entgeltpunkte_updated": "pension_entitlement_points_updated",
            "grundrente": {
                "gesamteinnahmen_aus_renten_für_einkommensberechnung_im_folgejahr_m": "pension_total_income_for_income_calculation_next_year_m",
            },
            "entgeltpunkte_updated": "pension_entitlement_points_updated",
            "wartezeit_15_jahre_erfüllt": "pension_waiting_period_15_years_fulfilled",
        },
        "arbeitslosen": {
            "mean_nettoeinkommen_für_bemessungsgrundlage_bei_arbeitslosigkeit_y": "mean_net_income_for_benefit_basis_in_case_of_unemployment_y",
            "beitrag": {
                "betrag_versicherter_m": "unemployment_insurance_contribution_m",
            }
        },
        "pflege": {"beitrag": {"betrag_gesamt_in_gleitzone_m": "long_term_care_insurance_contribution_total_in_transition_zone_m"}},
        "beiträge_gesamt_m": "social_insurance_contributions_total_m",
    },
    "kindergeld": {"betrag_m": "KG_betrag_m"},
    "bürgergeld": {"betrag_m_bg": "BG_betrag_m_bg"},
    "grundsicherung": {"im_alter": {"betrag_m_eg": "GS_betrag_m_eg"}},
    "wohngeld": {"betrag_m_wthh": "WG_betrag_m_wthh"},
    "kinderzuschlag": {
        "betrag_m_bg": "KiZ_betrag_m_bg",
    },
    "familie": {"alleinerziehend_fg": "single_parent_fg"},
    "elterngeld": {
        "betrag_m": "EG_betrag_m",
        "anrechenbarer_betrag_m": "EG_anrechenbarer_betrag_m",
        "mean_nettoeinkommen_für_bemessungsgrundlage_nach_geburt_m": "EG_mean_nettoeinkommen_für_bemessungsgrundlage_nach_geburt_m"
    },
    "unterhalt": {
        "tatsächlich_erhaltener_betrag_m": "unterhalt_tatsächlich_erhaltener_betrag_m",
        "kind_festgelegter_zahlbetrag_m": "unterhalt_kind_festgelegter_zahlbetrag_m",
    },
    "unterhaltsvorschuss": {
        "an_elternteil_auszuzahlender_betrag_m": "unterhaltsvorschuss_an_elternteil_auszuzahlender_betrag_m",
    },
}


# =============================================================================
# INPUT DATA MAPPER CONFIGURATION
# =============================================================================

MAPPER = {
    "alter": "age",
    "alter_monate": "alter_monate",
    "geburtsmonat": 1,
    "arbeitsstunden_w": "working_hours",
    "behinderungsgrad": "disability_grade",
    "schwerbehindert_grad_g": False,
    "geburtsjahr": "birth_year",
    "hh_id": "hh_id",
    "p_id": "p_id",
    "wohnort_ost_hh": "east_germany",
    "einnahmen": {
        "bruttolohn_m": 2000.0,
        "kapitalerträge_y": 0.0,
        "renten": {
            "betriebliche_altersvorsorge_m": 0.0,
            "geförderte_private_vorsorge_m": 0.0,
            "gesetzliche_m": 0.0,
            "sonstige_private_vorsorge_m": 0.0,
        },
    },
    "einkommensteuer": {
        "einkünfte": {
            "ist_hauptberuflich_selbstständig": False,
            "ist_selbstständig": "self_employed",
            "aus_gewerbebetrieb": {"betrag_m": "income_from_self_employment"},
            "aus_vermietung_und_verpachtung": {"betrag_m": "income_from_rent"},
            "aus_nichtselbstständiger_arbeit": {
                "bruttolohn_m": "income_from_employment"
            },
            "aus_forst_und_landwirtschaft": {
                "betrag_m": "income_from_forest_and_agriculture"
            },
            "aus_selbstständiger_arbeit": {"betrag_m": "income_from_self_employment"},
            "aus_kapitalvermögen": {"kapitalerträge_m": "income_from_capital"},
            "sonstige": {
                "alle_weiteren_y": 0.0,
                "ohne_renten_m": "income_from_other_sources",
                # "rente": {"ertragsanteil": 0.0},
                "renteneinkünfte_m": "pension_income",
            },
        },
        "abzüge": {
            "beitrag_private_rentenversicherung_m": "contribution_to_private_pension_insurance",  # noqa: E501
            "kinderbetreuungskosten_m": "childcare_expenses",
            "p_id_kinderbetreuungskostenträger": "person_that_pays_childcare_expenses",
        },
        "gemeinsam_veranlagt": "joint_taxation",
    },
    "lohnsteuer": {"steuerklasse": "lohnsteuer__steuerklasse"},
    "sozialversicherung": {
        "arbeitslosen": {
            # "betrag_m": 0.0
            "mean_nettoeinkommen_in_12_monaten_vor_arbeitslosigkeit_m": 2000.0,
            "arbeitssuchend": False,
            "monate_beitragspflichtig_versichert_in_letzten_30_monaten": 30,
            "monate_sozialversicherungspflichtiger_beschäftigung_in_letzten_5_jahren": 60,
            "monate_durchgängigen_bezugs_von_arbeitslosengeld": 0,
        },
        "rente": {
            "monat_renteneintritt": 1,
            "jahr_renteneintritt": "jahr_renteneintritt",
            "private_rente_betrag_m": "amount_private_pension_income",
            "monate_in_arbeitsunfähigkeit": 0,
            "krankheitszeiten_ab_16_bis_24_monate": 0.0,
            "monate_in_mutterschutz": 0,
            "monate_in_arbeitslosigkeit": 0,
            "monate_in_ausbildungssuche": 0,
            "monate_in_schulausbildung": 0,
            "monate_mit_bezug_entgeltersatzleistungen_wegen_arbeitslosigkeit": 0,
            "monate_geringfügiger_beschäftigung": 0,
            "kinderberücksichtigungszeiten_monate": 0,
            "pflegeberücksichtigungszeiten_monate": 0,
            "erwerbsminderung": {
                "voll_erwerbsgemindert": False,
                "teilweise_erwerbsgemindert": False,
            },
            "altersrente": {
                # "betrag_m": 0.0,
            },
            "grundrente": {
                "grundrentenzeiten_monate": 0,
                "bewertungszeiten_monate": 0,
                "gesamteinnahmen_aus_renten_vorjahr_m": 0.0,
                "mean_entgeltpunkte": 0.0,
                "bruttolohn_vorjahr_y": 20000.0,
                "einnahmen_aus_renten_vorjahr_y": 0.0,
                "einnahmen_aus_kapitalvermögen_vorvorjahr_y": 0.0,
                "einnahmen_aus_selbstständiger_arbeit_vorvorjahr_y": 0.0,
                "einnahmen_aus_vermietung_und_verpachtung_vorvorjahr_y": 0.0,
            },
            "bezieht_rente": False,
            "entgeltpunkte": 0.0,
            "pflichtbeitragsmonate": 0,
            "freiwillige_beitragsmonate": 0,
            "ersatzzeiten_monate": 0,
        },
        "kranken": {
            "beitrag": {"privat_versichert": "contribution_private_health_insurance"}
        },
        "pflege": {"beitrag": {"hat_kinder": "has_children"}},
    },
    "familie": {
        "alleinerziehend": "single_parent",
        "kind": "is_child",
        "p_id_ehepartner": "spouse_id",
        "p_id_elternteil_1": "parent_id_1",
        "p_id_elternteil_2": "parent_id_2",
    },
    "wohnen": {
        "bewohnt_eigentum_hh": False,
        "bruttokaltmiete_m_hh": 900.0,
        "heizkosten_m_hh": 150.0,
        "wohnfläche_hh": 80.0,
    },
    "kindergeld": {
        "in_ausbildung": "in_training",
        "p_id_empfänger": "id_recipient_child_allowance",
    },
    "vermögen": 0.0,
    "unterhalt": {
        "tatsächlich_erhaltener_betrag_m": 0.0,
        "anspruch_m": 0.0,
    },
    "elterngeld": {
        # "betrag_m": 0.0,
        # "anrechenbarer_betrag_m": 0.0,
        "zu_versteuerndes_einkommen_vorjahr_y_sn": 30000.0,
        "mean_nettoeinkommen_in_12_monaten_vor_geburt_m": 2000.0,
        "claimed": False,
        "bisherige_bezugsmonate": 0
    },
    "bürgergeld": {
        # "betrag_m_bg": 0.0,
        "p_id_einstandspartner": "bürgergeld__p_id_einstandspartner",
        "bezug_im_vorjahr": False,
    },
    "wohngeld": {
        # "betrag_m_wthh": 0.0,
        "mietstufe_hh": 3,
    },
    "kinderzuschlag": {
        # "betrag_m_bg": 0.0,
    },
}


# =============================================================================
# JAX UTILITIES
# =============================================================================

def sync_jax_if_needed(backend):
    """Force JAX synchronization to ensure all operations are complete."""
    if backend == "jax" and JAX_AVAILABLE:
        try:
            import jax
            # Force synchronization of all JAX operations
            jax.block_until_ready(jax.numpy.array([1.0]))
            print("  JAX operations synchronized")
        except ImportError:
            pass
        except Exception as e:
            print(f"  Warning: JAX sync failed: {e}")


def clear_jax_cache():
    """Clear JAX compilation cache to ensure clean state."""
    if JAX_AVAILABLE:
        try:
            import jax
            # Clear the JIT compilation cache
            jax.clear_caches()
            print("  JAX cache cleared")
        except Exception as e:
            print(f"  Warning: JAX cache clear failed: {e}")


# =============================================================================
# MEMORY TRACKING
# =============================================================================

def get_memory_usage_mb():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


class MemoryTracker:
    """Track peak memory usage during execution with continuous monitoring."""
    def __init__(self):
        self.peak_memory = 0
        self.process = psutil.Process(os.getpid())
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start continuous memory monitoring in background thread."""
        self.monitoring = True
        self.peak_memory = self.get_current_memory()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            self.update()
            time.sleep(0.01)  # Check every 10ms
    
    def get_current_memory(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def update(self):
        """Update peak memory if current usage is higher."""
        current = self.get_current_memory()
        if current > self.peak_memory:
            self.peak_memory = current
        return current
    
    def get_peak(self):
        """Get peak memory usage in MB."""
        return self.peak_memory


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def force_garbage_collection():
    """Force aggressive garbage collection between runs."""
    gc.collect()
    gc.collect()  # Run twice for good measure
    print("  Garbage collection completed")


def reset_session_state(backend):
    """Reset session state between different backend runs."""
    print(f"  Resetting session state for {backend} backend...")
    
    # Force garbage collection
    force_garbage_collection()
    
    # Clear JAX-specific state if switching to/from JAX
    if backend == "jax" or JAX_AVAILABLE:
        clear_jax_cache()
    
    # Add a small delay to let system settle
    time.sleep(0.5)


# =============================================================================
# COMMON DATASET SIZES
# =============================================================================

BENCHMARK_HOUSEHOLD_SIZES = [2**15-1, 2**15, 2**16, 2**17, 2**18, 2**19, 2**20]
PROFILE_HOUSEHOLD_SIZES = [2**15]  # Default for profiling: 32,768 households
BACKENDS = ["numpy", "jax"]