from app.dcf_engine import ForecastDrivers, run_dcf


def test_run_dcf_basic():
    drivers = ForecastDrivers(
        years=5,
        revenue_growth=[0.05]*5,
        ebit_margin=[0.2]*5,
        da_pct_rev=0.04,
        capex_pct_rev=0.05,
        wc_item_pct_rev=0.0,
    )
    res = run_dcf(
        base_revenue=100.0,
        drivers=drivers,
        tax_rate=0.21,
        wacc=0.09,
        terminal_growth=0.02,
        cash=10.0,
        debt=20.0,
        shares=10.0,
    )
    assert res.value_per_share is not None
    assert res.enterprise_value is not None
