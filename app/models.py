from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Meta(BaseModel):
    schemaVersion: str = "1.0"
    ticker: str
    companyName: Optional[str] = None
    currency: Optional[str] = None
    asOf: str
    dataProvider: str = "yahoo_finance"


class Market(BaseModel):
    price: Optional[float] = None
    shares: Optional[float] = None
    marketCap: Optional[float] = None
    beta: Optional[float] = None


class Assumptions(BaseModel):
    years: int = 5
    baseYear: Optional[str] = None
    terminalGrowth: float
    wacc: float
    riskFreeRate: float
    equityRiskPremium: float
    costOfDebt: float
    taxRate: float
    revenueGrowth: List[float]
    ebitMargin: List[float]
    daPctRevenue: float
    capexPctRevenue: float
    wcItemPctRevenue: float


class DCFBridge(BaseModel):
    enterpriseValue: Optional[float] = None
    cash: Optional[float] = None
    totalDebt: Optional[float] = None
    equityValue: Optional[float] = None


class DCFOutput(BaseModel):
    valuePerShare: Optional[float] = None
    bridge: DCFBridge
    pvOfFcfs: Optional[float] = None
    pvOfTerminal: Optional[float] = None


class Sensitivity(BaseModel):
    wacc: List[float]
    g: List[float]
    valuePerShareMatrix: List[List[Optional[float]]]


class Peer(BaseModel):
    ticker: str
    companyName: Optional[str] = None
    marketCap: Optional[float] = None
    currency: Optional[str] = None
    trailingPE: Optional[float] = None
    evToEbitda: Optional[float] = None
    evToSales: Optional[float] = None


class CompsSummary(BaseModel):
    peers: List[Peer] = Field(default_factory=list)
    multiples: Dict[str, Dict[str, Optional[float]]] = Field(default_factory=dict)
    impliedPriceRanges: Dict[str, Dict[str, Optional[float]]] = Field(default_factory=dict)


class FootballBar(BaseModel):
    method: str
    low: Optional[float] = None
    base: Optional[float] = None
    high: Optional[float] = None


class Narrative(BaseModel):
    drivers: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class DataQualityIssue(BaseModel):
    level: str  # info|warning|error
    message: str


class ValuationResponse(BaseModel):
    meta: Meta
    market: Market
    assumptions: Assumptions
    dcf: DCFOutput
    sensitivity: Sensitivity
    comps: CompsSummary
    football: List[FootballBar]
    narrative: Narrative
    dataQuality: List[DataQualityIssue] = Field(default_factory=list)
    raw: Dict[str, Any] = Field(default_factory=dict, description="Optional raw highlights for debugging")
