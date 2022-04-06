from ijcai2022nmmo import CompetitionConfig
import nmmo

class BaselineConfig(CompetitionConfig):
    NPOP = 1
    NENT = 8
    AGENTS = NPOP * [nmmo.Agent]