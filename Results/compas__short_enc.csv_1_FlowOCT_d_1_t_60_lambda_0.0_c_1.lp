\ Model FlowOCT
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  zeta[0,1,1] + zeta[0,2,1] + zeta[0,3,1] + zeta[1,1,1] + zeta[1,2,1]
   + zeta[1,3,1] + zeta[2,1,1] + zeta[2,2,1] + zeta[2,3,1] + zeta[3,1,1]
   + zeta[3,2,1] + zeta[3,3,1] + zeta[4,1,2] + zeta[4,2,2] + zeta[4,3,2]
   + zeta[5,1,1] + zeta[5,2,1] + zeta[5,3,1] + zeta[6,1,1] + zeta[6,2,1]
   + zeta[6,3,1] + zeta[7,1,1] + zeta[7,2,1] + zeta[7,3,1] + zeta[8,1,2]
   + zeta[8,2,2] + zeta[8,3,2] + zeta[9,1,1] + zeta[9,2,1] + zeta[9,3,1]
   - 30 Constant
Subject To
 R0: - zeta[0,1,1] - zeta[0,1,2] + z[0,1] - z[0,2] - z[0,3] = 0
 R1: - zeta[1,1,1] - zeta[1,1,2] + z[1,1] - z[1,2] - z[1,3] = 0
 R2: - zeta[2,1,1] - zeta[2,1,2] + z[2,1] - z[2,2] - z[2,3] = 0
 R3: - zeta[3,1,1] - zeta[3,1,2] + z[3,1] - z[3,2] - z[3,3] = 0
 R4: - zeta[4,1,1] - zeta[4,1,2] + z[4,1] - z[4,2] - z[4,3] = 0
 R5: - zeta[5,1,1] - zeta[5,1,2] + z[5,1] - z[5,2] - z[5,3] = 0
 R6: - zeta[6,1,1] - zeta[6,1,2] + z[6,1] - z[6,2] - z[6,3] = 0
 R7: - zeta[7,1,1] - zeta[7,1,2] + z[7,1] - z[7,2] - z[7,3] = 0
 R8: - zeta[8,1,1] - zeta[8,1,2] + z[8,1] - z[8,2] - z[8,3] = 0
 R9: - zeta[9,1,1] - zeta[9,1,2] + z[9,1] - z[9,2] - z[9,3] = 0
 R10: - b[1,age_factor.1] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] + z[0,2] <= 0
 R11: - b[1,two_year_recid.0] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] + z[1,2] <= 0
 R12: - b[1,two_year_recid.0] - b[1,age_factor.1] - b[1,age_factor.2]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] + z[2,2] <= 0
 R13: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,crime_factor.1] + z[3,2] <= 0
 R14: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,two_year_recid.0]
   - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.2]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] + z[4,2] <= 0
 R15: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] + z[5,2] <= 0
 R16: - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[6,2]
   <= 0
 R17: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.2]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] + z[7,2] <= 0
 R18: - b[1,two_year_recid.0] - b[1,age_factor.1] - b[1,age_factor.2]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[8,2]
   <= 0
 R19: - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[9,2]
   <= 0
 R20: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,race_factor.6]
   - b[1,crime_factor.1] + z[0,3] <= 0
 R21: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.1] - b[1,race_factor.2] - b[1,crime_factor.1] + z[1,3]
   <= 0
 R22: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.3] - b[1,race_factor.2] - b[1,crime_factor.1] + z[2,3]
   <= 0
 R23: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.6] + z[3,3]
   <= 0
 R24: - b[1,priors_count.3] - b[1,priors_count.4] - b[1,priors_count.5]
   - b[1,priors_count.6] - b[1,priors_count.7] - b[1,priors_count.8]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.1]
   - b[1,crime_factor.1] + z[4,3] <= 0
 R25: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.6]
   - b[1,crime_factor.1] + z[5,3] <= 0
 R26: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,age_factor.1] - b[1,race_factor.1] + z[6,3] <= 0
 R27: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.1]
   - b[1,crime_factor.1] + z[7,3] <= 0
 R28: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.3] - b[1,race_factor.2] + z[8,3] <= 0
 R29: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,age_factor.1] - b[1,race_factor.1] + z[9,3] <= 0
 R30: b[1,priors_count.1] + b[1,priors_count.2] + b[1,priors_count.3]
   + b[1,priors_count.4] + b[1,priors_count.5] + b[1,priors_count.6]
   + b[1,priors_count.7] + b[1,priors_count.8] + b[1,two_year_recid.0]
   + b[1,gender_factor.1] + b[1,age_factor.1] + b[1,age_factor.2]
   + b[1,age_factor.3] + b[1,race_factor.1] + b[1,race_factor.2]
   + b[1,race_factor.3] + b[1,race_factor.4] + b[1,race_factor.5]
   + b[1,race_factor.6] + b[1,crime_factor.1] + p[1] = 1
 R31: p[1] + p[2] = 1
 R32: p[1] + p[3] = 1
 R33: - beta[1,1] + zeta[0,1,1] <= 0
 R34: - beta[1,2] + zeta[0,1,2] <= 0
 R35: - beta[1,1] + zeta[1,1,1] <= 0
 R36: - beta[1,2] + zeta[1,1,2] <= 0
 R37: - beta[1,1] + zeta[2,1,1] <= 0
 R38: - beta[1,2] + zeta[2,1,2] <= 0
 R39: - beta[1,1] + zeta[3,1,1] <= 0
 R40: - beta[1,2] + zeta[3,1,2] <= 0
 R41: - beta[1,1] + zeta[4,1,1] <= 0
 R42: - beta[1,2] + zeta[4,1,2] <= 0
 R43: - beta[1,1] + zeta[5,1,1] <= 0
 R44: - beta[1,2] + zeta[5,1,2] <= 0
 R45: - beta[1,1] + zeta[6,1,1] <= 0
 R46: - beta[1,2] + zeta[6,1,2] <= 0
 R47: - beta[1,1] + zeta[7,1,1] <= 0
 R48: - beta[1,2] + zeta[7,1,2] <= 0
 R49: - beta[1,1] + zeta[8,1,1] <= 0
 R50: - beta[1,2] + zeta[8,1,2] <= 0
 R51: - beta[1,1] + zeta[9,1,1] <= 0
 R52: - beta[1,2] + zeta[9,1,2] <= 0
 R53: - beta[2,1] + zeta[0,2,1] <= 0
 R54: - beta[2,2] + zeta[0,2,2] <= 0
 R55: - beta[2,1] + zeta[1,2,1] <= 0
 R56: - beta[2,2] + zeta[1,2,2] <= 0
 R57: - beta[2,1] + zeta[2,2,1] <= 0
 R58: - beta[2,2] + zeta[2,2,2] <= 0
 R59: - beta[2,1] + zeta[3,2,1] <= 0
 R60: - beta[2,2] + zeta[3,2,2] <= 0
 R61: - beta[2,1] + zeta[4,2,1] <= 0
 R62: - beta[2,2] + zeta[4,2,2] <= 0
 R63: - beta[2,1] + zeta[5,2,1] <= 0
 R64: - beta[2,2] + zeta[5,2,2] <= 0
 R65: - beta[2,1] + zeta[6,2,1] <= 0
 R66: - beta[2,2] + zeta[6,2,2] <= 0
 R67: - beta[2,1] + zeta[7,2,1] <= 0
 R68: - beta[2,2] + zeta[7,2,2] <= 0
 R69: - beta[2,1] + zeta[8,2,1] <= 0
 R70: - beta[2,2] + zeta[8,2,2] <= 0
 R71: - beta[2,1] + zeta[9,2,1] <= 0
 R72: - beta[2,2] + zeta[9,2,2] <= 0
 R73: - beta[3,1] + zeta[0,3,1] <= 0
 R74: - beta[3,2] + zeta[0,3,2] <= 0
 R75: - beta[3,1] + zeta[1,3,1] <= 0
 R76: - beta[3,2] + zeta[1,3,2] <= 0
 R77: - beta[3,1] + zeta[2,3,1] <= 0
 R78: - beta[3,2] + zeta[2,3,2] <= 0
 R79: - beta[3,1] + zeta[3,3,1] <= 0
 R80: - beta[3,2] + zeta[3,3,2] <= 0
 R81: - beta[3,1] + zeta[4,3,1] <= 0
 R82: - beta[3,2] + zeta[4,3,2] <= 0
 R83: - beta[3,1] + zeta[5,3,1] <= 0
 R84: - beta[3,2] + zeta[5,3,2] <= 0
 R85: - beta[3,1] + zeta[6,3,1] <= 0
 R86: - beta[3,2] + zeta[6,3,2] <= 0
 R87: - beta[3,1] + zeta[7,3,1] <= 0
 R88: - beta[3,2] + zeta[7,3,2] <= 0
 R89: - beta[3,1] + zeta[8,3,1] <= 0
 R90: - beta[3,2] + zeta[8,3,2] <= 0
 R91: - beta[3,1] + zeta[9,3,1] <= 0
 R92: - beta[3,2] + zeta[9,3,2] <= 0
 R93: - p[1] + beta[1,1] + beta[1,2] = 0
 R94: - p[2] + beta[2,1] + beta[2,2] = 0
 R95: - p[3] + beta[3,1] + beta[3,2] = 0
 R96: zeta[0,2,1] + zeta[0,2,2] - z[0,2] = 0
 R97: zeta[1,2,1] + zeta[1,2,2] - z[1,2] = 0
 R98: zeta[2,2,1] + zeta[2,2,2] - z[2,2] = 0
 R99: zeta[3,2,1] + zeta[3,2,2] - z[3,2] = 0
 R100: zeta[4,2,1] + zeta[4,2,2] - z[4,2] = 0
 R101: zeta[5,2,1] + zeta[5,2,2] - z[5,2] = 0
 R102: zeta[6,2,1] + zeta[6,2,2] - z[6,2] = 0
 R103: zeta[7,2,1] + zeta[7,2,2] - z[7,2] = 0
 R104: zeta[8,2,1] + zeta[8,2,2] - z[8,2] = 0
 R105: zeta[9,2,1] + zeta[9,2,2] - z[9,2] = 0
 R106: zeta[0,3,1] + zeta[0,3,2] - z[0,3] = 0
 R107: zeta[1,3,1] + zeta[1,3,2] - z[1,3] = 0
 R108: zeta[2,3,1] + zeta[2,3,2] - z[2,3] = 0
 R109: zeta[3,3,1] + zeta[3,3,2] - z[3,3] = 0
 R110: zeta[4,3,1] + zeta[4,3,2] - z[4,3] = 0
 R111: zeta[5,3,1] + zeta[5,3,2] - z[5,3] = 0
 R112: zeta[6,3,1] + zeta[6,3,2] - z[6,3] = 0
 R113: zeta[7,3,1] + zeta[7,3,2] - z[7,3] = 0
 R114: zeta[8,3,1] + zeta[8,3,2] - z[8,3] = 0
 R115: zeta[9,3,1] + zeta[9,3,2] - z[9,3] = 0
 R116: 0.3333333333333333 zeta[0,1,2] + 0.3333333333333333 zeta[0,2,2]
   + 0.3333333333333333 zeta[0,3,2] - 0.3333333333333333 zeta[1,1,2]
   - 0.3333333333333333 zeta[1,2,2] - 0.3333333333333333 zeta[1,3,2]
   - 0.3333333333333333 zeta[2,1,2] - 0.3333333333333333 zeta[2,2,2]
   - 0.3333333333333333 zeta[2,3,2] + 0.3333333333333333 zeta[3,1,2]
   + 0.3333333333333333 zeta[3,2,2] + 0.3333333333333333 zeta[3,3,2]
   + 0.3333333333333333 zeta[5,1,2] + 0.3333333333333333 zeta[5,2,2]
   + 0.3333333333333333 zeta[5,3,2] - 0.3333333333333333 zeta[8,1,2]
   - 0.3333333333333333 zeta[8,2,2] - 0.3333333333333333 zeta[8,3,2]
   <= 0.02
 R117: 0.3333333333333333 zeta[0,1,2] + 0.3333333333333333 zeta[0,2,2]
   + 0.3333333333333333 zeta[0,3,2] - 0.3333333333333333 zeta[1,1,2]
   - 0.3333333333333333 zeta[1,2,2] - 0.3333333333333333 zeta[1,3,2]
   - 0.3333333333333333 zeta[2,1,2] - 0.3333333333333333 zeta[2,2,2]
   - 0.3333333333333333 zeta[2,3,2] + 0.3333333333333333 zeta[3,1,2]
   + 0.3333333333333333 zeta[3,2,2] + 0.3333333333333333 zeta[3,3,2]
   + 0.3333333333333333 zeta[5,1,2] + 0.3333333333333333 zeta[5,2,2]
   + 0.3333333333333333 zeta[5,3,2] - 0.3333333333333333 zeta[8,1,2]
   - 0.3333333333333333 zeta[8,2,2] - 0.3333333333333333 zeta[8,3,2]
   >= -0.02
 R118: 0.3333333333333333 zeta[0,1,2] + 0.3333333333333333 zeta[0,2,2]
   + 0.3333333333333333 zeta[0,3,2] + 0.3333333333333333 zeta[3,1,2]
   + 0.3333333333333333 zeta[3,2,2] + 0.3333333333333333 zeta[3,3,2]
   - 0.25 zeta[4,1,2] - 0.25 zeta[4,2,2] - 0.25 zeta[4,3,2]
   + 0.3333333333333333 zeta[5,1,2] + 0.3333333333333333 zeta[5,2,2]
   + 0.3333333333333333 zeta[5,3,2] - 0.25 zeta[6,1,2] - 0.25 zeta[6,2,2]
   - 0.25 zeta[6,3,2] - 0.25 zeta[7,1,2] - 0.25 zeta[7,2,2]
   - 0.25 zeta[7,3,2] - 0.25 zeta[9,1,2] - 0.25 zeta[9,2,2]
   - 0.25 zeta[9,3,2] <= 0.02
 R119: 0.3333333333333333 zeta[0,1,2] + 0.3333333333333333 zeta[0,2,2]
   + 0.3333333333333333 zeta[0,3,2] + 0.3333333333333333 zeta[3,1,2]
   + 0.3333333333333333 zeta[3,2,2] + 0.3333333333333333 zeta[3,3,2]
   - 0.25 zeta[4,1,2] - 0.25 zeta[4,2,2] - 0.25 zeta[4,3,2]
   + 0.3333333333333333 zeta[5,1,2] + 0.3333333333333333 zeta[5,2,2]
   + 0.3333333333333333 zeta[5,3,2] - 0.25 zeta[6,1,2] - 0.25 zeta[6,2,2]
   - 0.25 zeta[6,3,2] - 0.25 zeta[7,1,2] - 0.25 zeta[7,2,2]
   - 0.25 zeta[7,3,2] - 0.25 zeta[9,1,2] - 0.25 zeta[9,2,2]
   - 0.25 zeta[9,3,2] >= -0.02
 R120: 0.3333333333333333 zeta[1,1,2] + 0.3333333333333333 zeta[1,2,2]
   + 0.3333333333333333 zeta[1,3,2] + 0.3333333333333333 zeta[2,1,2]
   + 0.3333333333333333 zeta[2,2,2] + 0.3333333333333333 zeta[2,3,2]
   - 0.25 zeta[4,1,2] - 0.25 zeta[4,2,2] - 0.25 zeta[4,3,2]
   - 0.25 zeta[6,1,2] - 0.25 zeta[6,2,2] - 0.25 zeta[6,3,2]
   - 0.25 zeta[7,1,2] - 0.25 zeta[7,2,2] - 0.25 zeta[7,3,2]
   + 0.3333333333333333 zeta[8,1,2] + 0.3333333333333333 zeta[8,2,2]
   + 0.3333333333333333 zeta[8,3,2] - 0.25 zeta[9,1,2] - 0.25 zeta[9,2,2]
   - 0.25 zeta[9,3,2] <= 0.02
 R121: 0.3333333333333333 zeta[1,1,2] + 0.3333333333333333 zeta[1,2,2]
   + 0.3333333333333333 zeta[1,3,2] + 0.3333333333333333 zeta[2,1,2]
   + 0.3333333333333333 zeta[2,2,2] + 0.3333333333333333 zeta[2,3,2]
   - 0.25 zeta[4,1,2] - 0.25 zeta[4,2,2] - 0.25 zeta[4,3,2]
   - 0.25 zeta[6,1,2] - 0.25 zeta[6,2,2] - 0.25 zeta[6,3,2]
   - 0.25 zeta[7,1,2] - 0.25 zeta[7,2,2] - 0.25 zeta[7,3,2]
   + 0.3333333333333333 zeta[8,1,2] + 0.3333333333333333 zeta[8,2,2]
   + 0.3333333333333333 zeta[8,3,2] - 0.25 zeta[9,1,2] - 0.25 zeta[9,2,2]
   - 0.25 zeta[9,3,2] >= -0.02
Bounds
 Constant = 1
Binaries
 b[1,priors_count.1] b[1,priors_count.2] b[1,priors_count.3]
 b[1,priors_count.4] b[1,priors_count.5] b[1,priors_count.6]
 b[1,priors_count.7] b[1,priors_count.8] b[1,two_year_recid.0]
 b[1,gender_factor.1] b[1,age_factor.1] b[1,age_factor.2] b[1,age_factor.3]
 b[1,race_factor.1] b[1,race_factor.2] b[1,race_factor.3]
 b[1,race_factor.4] b[1,race_factor.5] b[1,race_factor.6]
 b[1,crime_factor.1] p[1] p[2] p[3]
End
