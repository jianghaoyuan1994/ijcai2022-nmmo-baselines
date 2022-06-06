# -*- coding: UTF-8 -*-
"""
 @Time    : 2022/5/19 13:38
 @Author  : 姜浩源
 @FileName: metrics.py.py
 @Software: PyCharm
"""
import nmmo
from nmmo import entity
from typing import List, Sequence, Callable

from ijcai2022nmmo import tasks


class Metrics(dict):
    @classmethod
    def names(cls) -> List[str]:
        return [
            "PlayerDefeats",
            "Equipment",
            "Exploration",
            "Foraging",
            "TimeAlive",
            "health"
        ]

    @classmethod
    def collect(cls, env: nmmo.Env, player: entity.Player) -> "Metrics":
        realm = env.realm
        forage = float(tasks.foraging(realm, player))
        explore = float(tasks.exploration(realm, player))
        kills = float(tasks.player_kills(realm, player))
        equipment = float(tasks.equipment(realm, player))
        health = float(player.resources.health.val)
        health_max = float(player.resources.health.max)
        timealive = float(player.history.timeAlive.val)
        return Metrics(
            **{
                "PlayerDefeats": kills,
                "Equipment": equipment,
                "Exploration": explore,
                "Foraging": forage,
                "TimeAlive": timealive,
                # "Task_all": float(tasks.All(realm, player)),
                "health": health,
                "health_max": health_max
            })

    @classmethod
    def sum(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(sum, metrices)

    @classmethod
    def max(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(max, metrices)

    @classmethod
    def min(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(min, metrices)

    @classmethod
    def avg(cls, metrices: Sequence["Metrics"]) -> "Metrics":
        return cls.reduce(lambda x: sum(x) / len(x) if len(x) else 0, metrices)

    @classmethod
    def reduce(cls, func: Callable,
               metrices: Sequence["Metrics"]) -> "Metrics":
        names = cls.names()
        values = [[m[name] for m in metrices] for name in names]
        return Metrics(**dict(zip(names, list(map(func, values)))))

