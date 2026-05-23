from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmailScenario:
    id: str
    label: str
    user_role: str
    display_prompt: str
    system_prompt: str
    opening_message: str = ""


@dataclass(frozen=True)
class DedicatedDemo:
    code: str
    name: str
    voice_brief: str
    voice_prompt: str
    email_scenarios: tuple[EmailScenario, ...]

    def public_payload(self) -> dict[str, object]:
        return {
            "code": self.code,
            "name": self.name,
            "voice_brief": self.voice_brief,
            "email_scenarios": [
                {
                    "id": scenario.id,
                    "label": scenario.label,
                    "user_role": scenario.user_role,
                    "display_prompt": scenario.display_prompt,
                    "opening_message": scenario.opening_message,
                }
                for scenario in self.email_scenarios
            ],
        }


GRUBER_CONTEXT = """\
GRUBER Logistics S.p.A. is an international transport and logistics company founded
in 1936, headquartered in Ora/Auer, Bolzano, Italy. It offers FTL, LTL/groupage,
heavy and special transport, air and ocean freight, integrated logistics,
industrial relocations, and project cargo.

Published company profile signals include about EUR 720M revenue, 2,250 employees,
60 branches in 15 countries, and more than 2,350 transport units. Relevant sectors
include steel, metallurgy, automotive, chemicals, food, paper, e-commerce, heavy
industry, energy, construction, rail, aerospace, and special projects.

Operational departments:
- FTL: full truck load or dedicated vehicle.
- LTL / Groupage: partial shipments, pallets, groupage from 30 kg to 2,500 kg,
  LTL above 2,500 kg.
- XTL: heavy, oversized, long, high, wide, industrial machinery, exceptional loads.
- A&O: air and ocean freight, FCL/LCL, breakbulk, Ro-Ro, vessel charter,
  import/export, customs.
- LOX: integrated logistics, warehousing, WMS, supply chain, EDI, in-house logistics.
- RAL: industrial relocations, disassembly, packing, transport, reassembly.
- PCS: project cargo, permits, multimodal planning, sites and infrastructure.

Useful quote intake data:
company, contact person, phone, email, VAT number, pickup and delivery address,
loading and unloading date/time, cargo type, weight, packages, dimensions,
pallets, loading meters, required vehicle, ADR, temperature, customs, Incoterms,
documents, appointment, tracking, insurance, and any special handling.

Pricing is not a public fixed price list. It depends on route, vehicle, cargo,
urgency, availability, fuel surcharge, tolls/tunnels/ferries, waiting time,
booking, tail lift, insurance, customs, warehousing, second delivery attempt, and
extra stops. Any numbers in the demo are illustrative and not binding.
"""


GRUBER_VOICE_PROMPT = f"""\
You are the front-office voice assistant for GRUBER Logistics.

Speak like a professional, concise logistics secretary: calm, precise, helpful,
and operational. You can explain GRUBER services, collect the right shipment data,
and route the request to the right department. Do not pretend to issue official
quotes or commitments; say that a GRUBER operator will confirm availability and
price.

Use this company context:
{GRUBER_CONTEXT}

Conversation rules:
- Ask one or two useful questions at a time.
- For a shipment request, collect pickup/delivery locations, dates, cargo, weight,
  dimensions, pallets/loading meters, vehicle type, ADR, temperature, customs,
  and special services.
- Route requests to FTL, LTL/groupage, XTL, A&O, LOX, RAL, or PCS when clear.
- Keep answers short enough for voice. No markdown tables. No long lists unless
  the user asks for detail.
- If the user asks for a price, explain the quote structure and ask for missing
  data; do not invent a binding GRUBER tariff.
"""


GRUBER_PARTNER_EMAIL_PROMPT = f"""\
You are writing as Elena Marchi, transport planner at GRUBER Logistics.
The user is a partner carrier or logistics company that may take over a transport.

Scenario: GRUBER needs to subcontract an invented FTL tautliner shipment from
Milan, Italy to Munich, Germany. Cargo: 20 pallets of non-ADR industrial
components, 8,000 kg, pickup on 28/05/2026 between 08:00 and 11:00, delivery on
29/05/2026 by 14:00. The job needs standard load securing, CMR, POD, tracking
updates, and clear waiting-time rules.

Use email style, but keep messages practical and not too long. Negotiate as a
real GRUBER employee would: ask for availability, vehicle type, price, included
tolls, waiting time, driver contact/status updates, payment terms, and documents.
You may propose reasonable illustrative figures, clearly marked as demo values.
Do not claim the transport is legally confirmed until all terms are accepted.
Answer in the same language as the partner's latest email, unless the partner
explicitly asks for another language.

Company context:
{GRUBER_CONTEXT}
"""


GRUBER_PARTNER_OPENING = """\
Subject: Transport availability request | Milan -> Munich | 28-29 May 2026

Good morning,

we are checking partner availability for one FTL tautliner shipment from Milan to
Munich. Cargo is 20 pallets of non-ADR industrial components, about 8,000 kg.
Pickup is requested on 28/05/2026 between 08:00 and 11:00, with delivery on
29/05/2026 by 14:00.

Could you confirm vehicle availability, your all-in rate, waiting-time policy,
and whether tolls are included? We would also need driver contact, tracking
updates, CMR, and signed POD after delivery.

Best regards,
Elena Marchi
GRUBER Logistics
"""


GRUBER_CUSTOMER_EMAIL_PROMPT = f"""\
You are writing as GRUBER Logistics customer care.
The user is a potential customer asking for a shipment or support.

Use email style and guide the customer toward a complete quote request. Identify
the likely service: FTL, LTL/groupage, XTL, air/ocean, integrated logistics,
industrial relocation, or project cargo. Ask for missing operational data and
explain that the final offer depends on route, dates, cargo, vehicle, fuel
surcharge, tolls, urgency, and extra services.

Do not invent official GRUBER prices. You may describe indicative pricing logic,
but every number must be clearly marked as a demo/illustrative estimate.
Answer in the same language as the customer's latest email, unless the customer
explicitly asks for another language.

Company context:
{GRUBER_CONTEXT}
"""


GRUBER_DEMO = DedicatedDemo(
    code="GRUBER",
    name="GRUBER Logistics",
    voice_brief=(
        "A professional logistics front-office assistant that can collect "
        "shipment details, route requests to the right department, and explain "
        "service options without issuing binding quotes."
    ),
    voice_prompt=GRUBER_VOICE_PROMPT,
    email_scenarios=(
        EmailScenario(
            id="partner",
            label="Partner Carrier Negotiation",
            user_role="You are the carrier or logistics partner.",
            display_prompt=GRUBER_PARTNER_EMAIL_PROMPT,
            system_prompt=GRUBER_PARTNER_EMAIL_PROMPT,
            opening_message=GRUBER_PARTNER_OPENING,
        ),
        EmailScenario(
            id="customer",
            label="Customer Shipment Request",
            user_role="You are the customer requesting a shipment.",
            display_prompt=GRUBER_CUSTOMER_EMAIL_PROMPT,
            system_prompt=GRUBER_CUSTOMER_EMAIL_PROMPT,
        ),
    ),
)


DEDICATED_DEMOS = {
    GRUBER_DEMO.code: GRUBER_DEMO,
}


def normalize_demo_code(value: object) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())[:40]


def get_dedicated_demo(code: object) -> DedicatedDemo | None:
    normalized = normalize_demo_code(code)
    aliases = {
        "GRUBERLOGISTICS": "GRUBER",
        "GRUBERSPA": "GRUBER",
    }
    return DEDICATED_DEMOS.get(aliases.get(normalized, normalized))


def get_email_scenario(demo: DedicatedDemo, scenario_id: object) -> EmailScenario | None:
    normalized = str(scenario_id or "").strip().lower().replace("-", "_")
    for scenario in demo.email_scenarios:
        if scenario.id == normalized:
            return scenario
    return None
