from dataclasses import dataclass, asdict
import random


@dataclass
class AttentionState:
    """stores all the state variables for each timestep"""
    t: int
    attention_level: float
    boredom: float
    fatigue: float
    last_event_type: str
    dopamine_hit: float
    volatility: float
    attention_liquidity: float
    attention_demand: float
    attention_imbalance: float
    regime: str


class AttentionEnv:
    """
    Main simulation environment. Models attention like a market where
    events (notifications, posts) act like orders that affect attention,
    boredom, fatigue, etc.
    """

    def __init__(
        self,
        max_time_steps: int = 1000,
        base_boredom_growth: float = 0.05,
        base_fatigue_growth: float = 0.02,
        dopamine_decay: float = 0.1,
        volatility_decay: float = 0.05,
    ):
        self.max_time_steps = max_time_steps
        self.base_boredom_growth = base_boredom_growth
        self.base_fatigue_growth = base_fatigue_growth
        self.dopamine_decay = dopamine_decay
        self.volatility_decay = volatility_decay

        # Adaptive algorithm ("market maker")
        self.algo_aggressiveness = 0.5
        self.notification_bias = 0.02
        self.high_impact_bias = 0.05

        self.reset()

    def reset(self):
        """reset to initial state"""
        self.t = 0

        self.state = AttentionState(
            t=0,
            attention_level=60.0,
            boredom=20.0,
            fatigue=10.0,
            last_event_type="none",
            dopamine_hit=0.0,
            volatility=5.0,
            attention_liquidity=70.0,
            attention_demand=1.0,
            attention_imbalance=0.0,
            regime="baseline"
        )

        self.history = []
        self._log_state()
        return self.state

    # utility functions
    def _clip(self, value, low=0.0, high=100.0):
        return max(low, min(high, value))

    def _log_state(self):
        self.history.append(asdict(self.state))

    # figure out which regime we're in based on current state
    def _compute_regime(self, attention, volatility, imbalance, boredom, fatigue):
        if attention > 60 and volatility < 10:
            return "engaged"
        if attention < 40 and (boredom > 50 or fatigue > 50):
            return "fatigued"
        if volatility > 20 and imbalance > 0:
            return "overstimulated"
        if imbalance > 0.5 and attention > 50:
            return "addictive_loop"
        if attention < 20:
            return "disengaged"
        return "baseline"

    # adjust the algorithm's aggressiveness based on what regime we're in
    # basically the platform backing off or pushing harder
    def _update_algorithm_behavior(self, regime):
        if regime == "engaged":
            self.algo_aggressiveness = max(0, self.algo_aggressiveness - 0.05)
            self.notification_bias = max(0.01, self.notification_bias - 0.005)
            self.high_impact_bias = max(0.03, self.high_impact_bias - 0.005)

        elif regime == "fatigued":
            self.algo_aggressiveness = max(0, self.algo_aggressiveness - 0.1)
            self.notification_bias = max(0.005, self.notification_bias - 0.01)
            self.high_impact_bias = max(0.02, self.high_impact_bias - 0.01)

        elif regime == "disengaged":
            self.algo_aggressiveness = min(1, self.algo_aggressiveness + 0.1)
            self.notification_bias = min(0.08, self.notification_bias + 0.02)
            self.high_impact_bias = min(0.08, self.high_impact_bias + 0.01)

        elif regime == "overstimulated":
            self.algo_aggressiveness = max(0, self.algo_aggressiveness - 0.07)
            self.notification_bias = max(0.01, self.notification_bias - 0.01)
            self.high_impact_bias = max(0.03, self.high_impact_bias - 0.01)

        elif regime == "addictive_loop":
            self.algo_aggressiveness = min(1, self.algo_aggressiveness + 0.05)
            self.notification_bias = min(0.1, self.notification_bias + 0.015)
            self.high_impact_bias = min(0.1, self.high_impact_bias + 0.015)

    # sample what type of event happens this timestep
    def _sample_event_type(self):
        p_notif = self.notification_bias * (0.5 + self.algo_aggressiveness)
        p_high = self.high_impact_bias * (0.5 + self.algo_aggressiveness)

        r = random.random()

        if r < p_notif:
            return "notification"
        elif r < p_notif + p_high:
            return "high_impact_post"
        else:
            return "baseline"

    def _event_dopamine_and_volatility(self, event_type: str):
        if event_type == "notification":
            return random.uniform(4, 8), random.uniform(3, 7)
        elif event_type == "high_impact_post":
            return random.uniform(3, 6), random.uniform(2, 5)
        else:
            return random.uniform(0, 2), random.uniform(0, 1.5)

    def _event_demand(self, event_type: str):
        if event_type == "notification":
            return random.uniform(5, 10)
        elif event_type == "high_impact_post":
            return random.uniform(2, 6)
        else:
            return random.uniform(0.5, 1.5)

    def step(self):
        """run one timestep of the simulation"""
        if self.t >= self.max_time_steps:
            return self.state, True

        self.t += 1
        event_type = self._sample_event_type()
        dopamine_hit, vol_shock = self._event_dopamine_and_volatility(event_type)

        s = self.state

        # boredom and fatigue naturally grow over time
        boredom = s.boredom + self.base_boredom_growth
        fatigue = s.fatigue + self.base_fatigue_growth

        # dopamine reduces boredom and boosts attention
        boredom -= 0.3 * dopamine_hit
        attention_boost = dopamine_hit * 1.5

        # boredom and fatigue drag attention down
        attention_drag = 0.1 * boredom + 0.2 * fatigue

        # update volatility
        volatility = s.volatility + vol_shock
        volatility *= (1 - self.volatility_decay)

        # calculate demand and liquidity (market-style)
        demand = self._event_demand(event_type)
        liquidity = max(0.0, 100.0 - boredom - fatigue)

        # compute imbalance
        if demand + liquidity > 0:
            imbalance = (demand - liquidity) / (demand + liquidity)
        else:
            imbalance = 0.0

        # update attention level
        attention_level = s.attention_level + attention_boost - attention_drag
        attention_level = self._clip(attention_level)
        boredom = self._clip(boredom)
        fatigue = self._clip(fatigue)

        regime = self._compute_regime(attention_level, volatility, imbalance, boredom, fatigue)

        # let the algorithm adapt its behavior
        self._update_algorithm_behavior(regime)

        # save the new state
        self.state = AttentionState(
            t=self.t,
            attention_level=attention_level,
            boredom=boredom,
            fatigue=fatigue,
            last_event_type=event_type,
            dopamine_hit=dopamine_hit,
            volatility=volatility,
            attention_liquidity=liquidity,
            attention_demand=demand,
            attention_imbalance=imbalance,
            regime=regime,
        )

        self._log_state()
        return self.state, self.t >= self.max_time_steps

    def run(self):
        """run the full simulation until max timesteps"""
        while True:
            _, done = self.step()
            if done:
                break
        return self.history
