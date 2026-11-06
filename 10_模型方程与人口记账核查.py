"""Synthetic equation checks for model v0.1; NOT an equilibrium solver.

Run: python 10_模型方程与人口记账核查.py
Only Python's standard library is needed. All numbers below are arbitrary test
fixtures, not estimates, calibration targets, Chinese data, or policy effects.
"""

from dataclasses import dataclass
from math import exp, fsum, isclose, isfinite, log
import json
import unittest


def logsumexp(values, scale=1.0):
    """Centered type-I extreme-value inclusive value (equations 11, 13)."""
    if not isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    values = list(values)
    if any(v != v or v == float("inf") for v in values):
        raise ValueError("NaN and positive infinity are invalid option values")
    feasible = [v for v in values if v != -float("inf")]
    if not feasible:
        raise ValueError("no feasible option")
    peak = max(feasible)
    return peak + scale * log(fsum(exp((v - peak) / scale) for v in feasible))


def probabilities(values, scale=1.0):
    """Stable probabilities; infeasible options have exactly zero probability."""
    values = list(values)
    logsumexp(values, scale)  # also validates inputs
    peak = max(values)
    weights = [exp((v - peak) / scale) for v in values]
    total = fsum(weights)
    return [w / total for w in weights]


def nested_choices(birth_values_by_destination, migration_costs,
                   birth_scale=1.0, migration_scale=1.0):
    """Location first, birth shock second; values already include continuation.

    This function does not solve intratemporal controls or a Bellman equation.
    A wholly infeasible destination is removed before computing the outer logit.
    """
    if len(birth_values_by_destination) != len(migration_costs):
        raise ValueError("destination lengths differ")
    inclusive, birth_probs = [], []
    for values, cost in zip(birth_values_by_destination, migration_costs):
        if not isfinite(cost) or cost < 0:
            raise ValueError("migration cost must be finite and nonnegative")
        if not values or all(v == -float("inf") for v in values):
            inclusive.append(-float("inf"))
            birth_probs.append([0.0] * len(values))
        else:
            inclusive.append(logsumexp(values, birth_scale) - cost)
            birth_probs.append(probabilities(values, birth_scale))
    move_probs = probabilities(inclusive, migration_scale)
    joint = [[move_probs[j] * v for v in row]
             for j, row in enumerate(birth_probs)]
    return logsumexp(inclusive, migration_scale), move_probs, birth_probs, joint


def ces_trade(wages, productivity, firm_masses, iceberg, elasticity):
    """Prices and shares for (15)-(17); row=origin, column=destination."""
    size = len(wages)
    if elasticity <= 1 or not isfinite(elasticity):
        raise ValueError("elasticity must exceed one")
    if not (size and len(productivity) == len(firm_masses) == len(iceberg) == size):
        raise ValueError("inconsistent dimensions")
    if any(len(row) != size for row in iceberg):
        raise ValueError("iceberg matrix must be square")
    if any(not isfinite(v) or v <= 0
           for vector in [wages, productivity, firm_masses] for v in vector):
        raise ValueError("wages, productivity and firm mass must be positive")
    if any(not isfinite(v) or v < 1 for row in iceberg for v in row):
        raise ValueError("iceberg costs must be at least one")
    markup = elasticity / (elasticity - 1)
    producer = [markup * w / a for w, a in zip(wages, productivity)]
    prices, shares = [], [[0.0] * size for _ in range(size)]
    for j in range(size):
        terms = [log(firm_masses[i]) + (1 - elasticity)
                 * log(iceberg[i][j] * producer[i]) for i in range(size)]
        denom = logsumexp(terms)
        prices.append(exp(denom / (1 - elasticity)))
        for i, value in enumerate(terms):
            shares[i][j] = exp(value - denom)
    return producer, prices, shares


@dataclass(frozen=True)
class Family:
    """A household (two adults), plus children aged 1..19 at start of year."""
    age: int
    parity: int
    children: tuple

    def validate(self):
        if not 20 <= self.age <= 84 or len(self.children) != 19:
            raise ValueError("invalid age or child vector")
        if self.parity < 0 or int(self.parity) != self.parity:
            raise ValueError("parity must be a nonnegative integer")
        if any(int(k) != k or k < 0 for k in self.children):
            raise ValueError("children must be nonnegative integers")
        if sum(self.children) > self.parity:
            raise ValueError("dependent children cannot exceed lifetime parity")
        for age, count in enumerate(self.children, start=1):
            if count and not 20 <= self.age - age <= 44:
                raise ValueError("child age inconsistent with baseline fertility ages")


def advance_family(family, birth, maximum_parity=4):
    """Equations (9), (28)-(29).

    Returns (surviving parent household or None, entrants in CHILD PERSON units).
    New household mass is entrants/2; there are two adults in every new household.
    """
    family.validate()
    if birth not in (0, 1):
        raise ValueError("birth must be zero or one")
    if birth and not (20 <= family.age <= 44 and family.parity < maximum_parity):
        raise ValueError("birth outside feasible set")
    entrants = family.children[-1]
    if family.age == 84:
        if any(family.children) or birth:
            raise ValueError("orphan case is outside the baseline")
        return None, entrants
    updated = Family(family.age + 1, family.parity + birth,
                     (birth,) + family.children[:-1])
    updated.validate()
    return updated, entrants


def persons(distribution):
    """Count each adult and dependent once; mass counts HOUSEHOLDS."""
    return fsum(mass * (2 + sum(family.children))
                for family, mass in distribution)


def advance_distribution(distribution, birth_choices):
    """Demographic accounting only: one location/type, no economic decisions."""
    if len(distribution) != len(birth_choices):
        raise ValueError("choice lengths differ")
    next_distribution, births, exiting_people, entrants = [], 0.0, 0.0, 0.0
    for (family, mass), birth in zip(distribution, birth_choices):
        if mass < 0 or not isfinite(mass):
            raise ValueError("invalid household mass")
        updated, child_entrants = advance_family(family, birth)
        births += mass * birth
        entrants += mass * child_entrants
        if updated is None:
            exiting_people += 2 * mass
        else:
            next_distribution.append((updated, mass))
    if entrants:
        next_distribution.append((Family(20, 0, (0,) * 19), entrants / 2))
    return next_distribution, births, exiting_people, entrants


class ModelChecks(unittest.TestCase):
    def test_stable_choice_and_infeasibility(self):
        values = [10000.0, 10001.0, -float("inf")]
        probs = probabilities(values, 0.7)
        self.assertAlmostEqual(sum(probs), 1)
        self.assertEqual(probs[-1], 0)
        shifted = probabilities([v + 50000 for v in values], 0.7)
        for a, b in zip(probs, shifted):
            self.assertAlmostEqual(a, b)
        self.assertEqual(logsumexp([12.0]), 12.0)
        with self.assertRaises(ValueError):
            probabilities([-float("inf")])

    def test_nested_probability_accounting(self):
        _, move, conditional, joint = nested_choices(
            [[1.0, 1.4], [1.2, -float("inf")], [-float("inf")]],
            [0.0, 0.2, 0.1], birth_scale=0.4, migration_scale=0.8)
        self.assertAlmostEqual(sum(move), 1)
        self.assertAlmostEqual(fsum(fsum(row) for row in joint), 1)
        self.assertEqual(move[2], 0)
        self.assertEqual(conditional[1][1], 0)
        self.assertGreater(conditional[0][1], 0.5)

    def test_ces_shares_prices_and_derivative(self):
        wage = [1.0, 1.3]
        prod = [1.0, 1.2]
        firms = [1.0, 1.4]
        trade = [[1.0, 1.2], [1.4, 1.0]]
        _, prices, shares = ces_trade(wage, prod, firms, trade, 4.0)
        for j in range(2):
            self.assertAlmostEqual(fsum(shares[i][j] for i in range(2)), 1)
        _, scaled, unchanged = ces_trade([3 * w for w in wage], prod, firms, trade, 4)
        for j in range(2):
            self.assertAlmostEqual(scaled[j], 3 * prices[j])
            for i in range(2):
                self.assertAlmostEqual(unchanged[i][j], shares[i][j])
        # d log P_j / d log p_i = expenditure share, holding other prices fixed.
        step = 1e-6
        _, perturbed, _ = ces_trade([wage[0] * exp(step), wage[1]], prod,
                                     firms, trade, 4)
        for j in range(2):
            derivative = log(perturbed[j] / prices[j]) / step
            self.assertAlmostEqual(derivative, shares[0][j], places=5)

    def test_birth_enters_adulthood_after_twenty_years(self):
        distribution = [(Family(20, 0, (0,) * 19), 1.0)]
        first_entry = None
        for year in range(66):
            before = persons(distribution)
            choices = [1 if year == 0 else 0 for _ in distribution]
            after, births, exits, entrants = advance_distribution(distribution, choices)
            self.assertAlmostEqual(persons(after), before + births - exits)
            if entrants and first_entry is None:
                first_entry = year + 1
                self.assertEqual(entrants, 1.0)
                self.assertAlmostEqual(sum(m for _, m in after), 1.5)
            if year == 0:
                self.assertEqual(sum(m for _, m in after), 1.0)
                self.assertEqual(persons(after), 3.0)
            distribution = after
        self.assertEqual(first_entry, 20)

    def test_national_migration_conserves_counts(self):
        origin_masses = [3.0, 7.0]
        migration = [[0.8, 0.2], [0.1, 0.9]]
        destination = [fsum(origin_masses[i] * migration[i][j]
                            for i in range(2)) for j in range(2)]
        self.assertAlmostEqual(sum(destination), sum(origin_masses))

    def test_child_ages_parity_and_feasibility(self):
        with self.assertRaises(ValueError):
            Family(20, 1, (0,) * 18 + (1,)).validate()
        with self.assertRaises(ValueError):
            advance_family(Family(45, 0, (0,) * 19), 1)
        parent = Family(39, 1, (0,) * 18 + (1,))
        updated, entrants = advance_family(parent, 0)
        self.assertEqual(updated.parity, 1)
        self.assertEqual(sum(updated.children), 0)
        self.assertEqual(entrants, 1)

    def test_resource_fiscal_and_profit_accounting(self):
        # All fixtures are synthetic. Construct an accounting-consistent economy;
        # this does NOT assert household optimality or solve regional equilibrium.
        wage, revenue, epsilon = 1.0, 100.0, 4.0
        markup = epsilon / (epsilon - 1)
        goods_labor = revenue / (markup * wage)
        profit = revenue / epsilon
        childcare_quantity, childcare_productivity, scarcity_rent = 10.0, 2.0, 0.3
        childcare_price = wage / childcare_productivity + scarcity_rent
        childcare_labor = childcare_quantity / childcare_productivity
        labor_income = wage * (goods_labor + childcare_labor)
        rent_revenue = 20.0
        newborn_transfers, pension, subsidy = 2.0, 6.0, 0.25
        government_spending = (newborn_transfers + pension
                               + subsidy * childcare_price * childcare_quantity)
        tax = government_spending / labor_income
        fund = rent_revenue + profit + scarcity_rent * childcare_quantity
        disposable = (1 - tax) * labor_income + newborn_transfers + pension + fund
        goods_expenditure = (disposable - rent_revenue
                             - (1 - subsidy) * childcare_price * childcare_quantity)
        self.assertTrue(0 <= tax < 1)
        self.assertAlmostEqual(tax * labor_income, government_spending)
        self.assertAlmostEqual(goods_expenditure, revenue)
        self.assertAlmostEqual(goods_labor * wage + profit, revenue)
        # Forgetting childcare wages would falsely count them as extra goods output.
        self.assertGreater(labor_income + profit, goods_expenditure)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ModelChecks))
    print(json.dumps({
        "status": "passed" if result.wasSuccessful() else "failed",
        "tests_run": result.testsRun,
        "real_data_used": False,
        "estimated_parameters": False,
        "general_equilibrium_solved": False,
        "scope": "synthetic choice, CES, population and fiscal accounting checks",
    }, ensure_ascii=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
