\ Model FlowOCT
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  zeta[9,1,1] + zeta[9,2,1] + zeta[9,3,1] + zeta[9,4,1] + zeta[9,5,1]
   + zeta[9,6,1] + zeta[9,7,1] + zeta[4,1,2] + zeta[4,2,2] + zeta[4,3,2]
   + zeta[4,4,2] + zeta[4,5,2] + zeta[4,6,2] + zeta[4,7,2] + zeta[2,1,1]
   + zeta[2,2,1] + zeta[2,3,1] + zeta[2,4,1] + zeta[2,5,1] + zeta[2,6,1]
   + zeta[2,7,1] + zeta[8,1,2] + zeta[8,2,2] + zeta[8,3,2] + zeta[8,4,2]
   + zeta[8,5,2] + zeta[8,6,2] + zeta[8,7,2] - 28 Constant
Subject To
 R0: - zeta[9,1,1] - zeta[9,1,2] + z[9,1] - z[9,2] - z[9,3] = 0
 R1: - zeta[4,1,1] - zeta[4,1,2] + z[4,1] - z[4,2] - z[4,3] = 0
 R2: - zeta[2,1,1] - zeta[2,1,2] + z[2,1] - z[2,2] - z[2,3] = 0
 R3: - zeta[8,1,1] - zeta[8,1,2] + z[8,1] - z[8,2] - z[8,3] = 0
 R4: - zeta[9,2,1] - zeta[9,2,2] + z[9,2] - z[9,4] - z[9,5] = 0
 R5: - zeta[4,2,1] - zeta[4,2,2] + z[4,2] - z[4,4] - z[4,5] = 0
 R6: - zeta[2,2,1] - zeta[2,2,2] + z[2,2] - z[2,4] - z[2,5] = 0
 R7: - zeta[8,2,1] - zeta[8,2,2] + z[8,2] - z[8,4] - z[8,5] = 0
 R8: - zeta[9,3,1] - zeta[9,3,2] + z[9,3] - z[9,6] - z[9,7] = 0
 R9: - zeta[4,3,1] - zeta[4,3,2] + z[4,3] - z[4,6] - z[4,7] = 0
 R10: - zeta[2,3,1] - zeta[2,3,2] + z[2,3] - z[2,6] - z[2,7] = 0
 R11: - zeta[8,3,1] - zeta[8,3,2] + z[8,3] - z[8,6] - z[8,7] = 0
 R12: - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[9,2]
   <= 0
 R13: - b[2,gender_factor.1] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1] + z[9,4]
   <= 0
 R14: - b[3,gender_factor.1] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1] + z[9,6]
   <= 0
 R15: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,two_year_recid.0]
   - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.2]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] + z[4,2] <= 0
 R16: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,two_year_recid.0]
   - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.2]
   - b[2,race_factor.3] - b[2,race_factor.4] - b[2,race_factor.5]
   - b[2,race_factor.6] + z[4,4] <= 0
 R17: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,two_year_recid.0]
   - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.2]
   - b[3,race_factor.3] - b[3,race_factor.4] - b[3,race_factor.5]
   - b[3,race_factor.6] + z[4,6] <= 0
 R18: - b[1,two_year_recid.0] - b[1,age_factor.1] - b[1,age_factor.2]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] + z[2,2] <= 0
 R19: - b[2,two_year_recid.0] - b[2,age_factor.1] - b[2,age_factor.2]
   - b[2,race_factor.1] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] + z[2,4] <= 0
 R20: - b[3,two_year_recid.0] - b[3,age_factor.1] - b[3,age_factor.2]
   - b[3,race_factor.1] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] + z[2,6] <= 0
 R21: - b[1,two_year_recid.0] - b[1,age_factor.1] - b[1,age_factor.2]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[8,2]
   <= 0
 R22: - b[2,two_year_recid.0] - b[2,age_factor.1] - b[2,age_factor.2]
   - b[2,race_factor.1] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1] + z[8,4]
   <= 0
 R23: - b[3,two_year_recid.0] - b[3,age_factor.1] - b[3,age_factor.2]
   - b[3,race_factor.1] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1] + z[8,6]
   <= 0
 R24: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,age_factor.1] - b[1,race_factor.1] + z[9,3] <= 0
 R25: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,age_factor.1] - b[2,race_factor.1] + z[9,5] <= 0
 R26: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,age_factor.1] - b[3,race_factor.1] + z[9,7] <= 0
 R27: - b[1,priors_count.3] - b[1,priors_count.4] - b[1,priors_count.5]
   - b[1,priors_count.6] - b[1,priors_count.7] - b[1,priors_count.8]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.1]
   - b[1,crime_factor.1] + z[4,3] <= 0
 R28: - b[2,priors_count.3] - b[2,priors_count.4] - b[2,priors_count.5]
   - b[2,priors_count.6] - b[2,priors_count.7] - b[2,priors_count.8]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.1]
   - b[2,crime_factor.1] + z[4,5] <= 0
 R29: - b[3,priors_count.3] - b[3,priors_count.4] - b[3,priors_count.5]
   - b[3,priors_count.6] - b[3,priors_count.7] - b[3,priors_count.8]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.1]
   - b[3,crime_factor.1] + z[4,7] <= 0
 R30: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.3] - b[1,race_factor.2] - b[1,crime_factor.1] + z[2,3]
   <= 0
 R31: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.3] - b[2,race_factor.2] - b[2,crime_factor.1] + z[2,5]
   <= 0
 R32: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.3] - b[3,race_factor.2] - b[3,crime_factor.1] + z[2,7]
   <= 0
 R33: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.3] - b[1,race_factor.2] + z[8,3] <= 0
 R34: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.3] - b[2,race_factor.2] + z[8,5] <= 0
 R35: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.3] - b[3,race_factor.2] + z[8,7] <= 0
 R36: b[1,priors_count.1] + b[1,priors_count.2] + b[1,priors_count.3]
   + b[1,priors_count.4] + b[1,priors_count.5] + b[1,priors_count.6]
   + b[1,priors_count.7] + b[1,priors_count.8] + b[1,two_year_recid.0]
   + b[1,gender_factor.1] + b[1,age_factor.1] + b[1,age_factor.2]
   + b[1,age_factor.3] + b[1,race_factor.1] + b[1,race_factor.2]
   + b[1,race_factor.3] + b[1,race_factor.4] + b[1,race_factor.5]
   + b[1,race_factor.6] + b[1,crime_factor.1] + p[1] = 1
 R37: b[2,priors_count.1] + b[2,priors_count.2] + b[2,priors_count.3]
   + b[2,priors_count.4] + b[2,priors_count.5] + b[2,priors_count.6]
   + b[2,priors_count.7] + b[2,priors_count.8] + b[2,two_year_recid.0]
   + b[2,gender_factor.1] + b[2,age_factor.1] + b[2,age_factor.2]
   + b[2,age_factor.3] + b[2,race_factor.1] + b[2,race_factor.2]
   + b[2,race_factor.3] + b[2,race_factor.4] + b[2,race_factor.5]
   + b[2,race_factor.6] + b[2,crime_factor.1] + p[1] + p[2] = 1
 R38: b[3,priors_count.1] + b[3,priors_count.2] + b[3,priors_count.3]
   + b[3,priors_count.4] + b[3,priors_count.5] + b[3,priors_count.6]
   + b[3,priors_count.7] + b[3,priors_count.8] + b[3,two_year_recid.0]
   + b[3,gender_factor.1] + b[3,age_factor.1] + b[3,age_factor.2]
   + b[3,age_factor.3] + b[3,race_factor.1] + b[3,race_factor.2]
   + b[3,race_factor.3] + b[3,race_factor.4] + b[3,race_factor.5]
   + b[3,race_factor.6] + b[3,crime_factor.1] + p[1] + p[3] = 1
 R39: p[1] + p[2] + p[4] = 1
 R40: p[1] + p[2] + p[5] = 1
 R41: p[1] + p[3] + p[6] = 1
 R42: p[1] + p[3] + p[7] = 1
 R43: - beta[1,1] + zeta[9,1,1] <= 0
 R44: - beta[1,2] + zeta[9,1,2] <= 0
 R45: - beta[1,1] + zeta[4,1,1] <= 0
 R46: - beta[1,2] + zeta[4,1,2] <= 0
 R47: - beta[1,1] + zeta[2,1,1] <= 0
 R48: - beta[1,2] + zeta[2,1,2] <= 0
 R49: - beta[1,1] + zeta[8,1,1] <= 0
 R50: - beta[1,2] + zeta[8,1,2] <= 0
 R51: - beta[2,1] + zeta[9,2,1] <= 0
 R52: - beta[2,2] + zeta[9,2,2] <= 0
 R53: - beta[2,1] + zeta[4,2,1] <= 0
 R54: - beta[2,2] + zeta[4,2,2] <= 0
 R55: - beta[2,1] + zeta[2,2,1] <= 0
 R56: - beta[2,2] + zeta[2,2,2] <= 0
 R57: - beta[2,1] + zeta[8,2,1] <= 0
 R58: - beta[2,2] + zeta[8,2,2] <= 0
 R59: - beta[3,1] + zeta[9,3,1] <= 0
 R60: - beta[3,2] + zeta[9,3,2] <= 0
 R61: - beta[3,1] + zeta[4,3,1] <= 0
 R62: - beta[3,2] + zeta[4,3,2] <= 0
 R63: - beta[3,1] + zeta[2,3,1] <= 0
 R64: - beta[3,2] + zeta[2,3,2] <= 0
 R65: - beta[3,1] + zeta[8,3,1] <= 0
 R66: - beta[3,2] + zeta[8,3,2] <= 0
 R67: - beta[4,1] + zeta[9,4,1] <= 0
 R68: - beta[4,2] + zeta[9,4,2] <= 0
 R69: - beta[4,1] + zeta[4,4,1] <= 0
 R70: - beta[4,2] + zeta[4,4,2] <= 0
 R71: - beta[4,1] + zeta[2,4,1] <= 0
 R72: - beta[4,2] + zeta[2,4,2] <= 0
 R73: - beta[4,1] + zeta[8,4,1] <= 0
 R74: - beta[4,2] + zeta[8,4,2] <= 0
 R75: - beta[5,1] + zeta[9,5,1] <= 0
 R76: - beta[5,2] + zeta[9,5,2] <= 0
 R77: - beta[5,1] + zeta[4,5,1] <= 0
 R78: - beta[5,2] + zeta[4,5,2] <= 0
 R79: - beta[5,1] + zeta[2,5,1] <= 0
 R80: - beta[5,2] + zeta[2,5,2] <= 0
 R81: - beta[5,1] + zeta[8,5,1] <= 0
 R82: - beta[5,2] + zeta[8,5,2] <= 0
 R83: - beta[6,1] + zeta[9,6,1] <= 0
 R84: - beta[6,2] + zeta[9,6,2] <= 0
 R85: - beta[6,1] + zeta[4,6,1] <= 0
 R86: - beta[6,2] + zeta[4,6,2] <= 0
 R87: - beta[6,1] + zeta[2,6,1] <= 0
 R88: - beta[6,2] + zeta[2,6,2] <= 0
 R89: - beta[6,1] + zeta[8,6,1] <= 0
 R90: - beta[6,2] + zeta[8,6,2] <= 0
 R91: - beta[7,1] + zeta[9,7,1] <= 0
 R92: - beta[7,2] + zeta[9,7,2] <= 0
 R93: - beta[7,1] + zeta[4,7,1] <= 0
 R94: - beta[7,2] + zeta[4,7,2] <= 0
 R95: - beta[7,1] + zeta[2,7,1] <= 0
 R96: - beta[7,2] + zeta[2,7,2] <= 0
 R97: - beta[7,1] + zeta[8,7,1] <= 0
 R98: - beta[7,2] + zeta[8,7,2] <= 0
 R99: - p[1] + beta[1,1] + beta[1,2] = 0
 R100: - p[2] + beta[2,1] + beta[2,2] = 0
 R101: - p[3] + beta[3,1] + beta[3,2] = 0
 R102: - p[4] + beta[4,1] + beta[4,2] = 0
 R103: - p[5] + beta[5,1] + beta[5,2] = 0
 R104: - p[6] + beta[6,1] + beta[6,2] = 0
 R105: - p[7] + beta[7,1] + beta[7,2] = 0
 R106: zeta[9,4,1] + zeta[9,4,2] - z[9,4] = 0
 R107: zeta[4,4,1] + zeta[4,4,2] - z[4,4] = 0
 R108: zeta[2,4,1] + zeta[2,4,2] - z[2,4] = 0
 R109: zeta[8,4,1] + zeta[8,4,2] - z[8,4] = 0
 R110: zeta[9,5,1] + zeta[9,5,2] - z[9,5] = 0
 R111: zeta[4,5,1] + zeta[4,5,2] - z[4,5] = 0
 R112: zeta[2,5,1] + zeta[2,5,2] - z[2,5] = 0
 R113: zeta[8,5,1] + zeta[8,5,2] - z[8,5] = 0
 R114: zeta[9,6,1] + zeta[9,6,2] - z[9,6] = 0
 R115: zeta[4,6,1] + zeta[4,6,2] - z[4,6] = 0
 R116: zeta[2,6,1] + zeta[2,6,2] - z[2,6] = 0
 R117: zeta[8,6,1] + zeta[8,6,2] - z[8,6] = 0
 R118: zeta[9,7,1] + zeta[9,7,2] - z[9,7] = 0
 R119: zeta[4,7,1] + zeta[4,7,2] - z[4,7] = 0
 R120: zeta[2,7,1] + zeta[2,7,2] - z[2,7] = 0
 R121: zeta[8,7,1] + zeta[8,7,2] - z[8,7] = 0
 R122: 0.5 zeta[9,1,2] + 0.5 zeta[9,2,2] + 0.5 zeta[9,3,2]
   + 0.5 zeta[9,4,2] + 0.5 zeta[9,5,2] + 0.5 zeta[9,6,2] + 0.5 zeta[9,7,2]
   + 0.5 zeta[4,1,2] + 0.5 zeta[4,2,2] + 0.5 zeta[4,3,2] + 0.5 zeta[4,4,2]
   + 0.5 zeta[4,5,2] + 0.5 zeta[4,6,2] + 0.5 zeta[4,7,2] - 0.5 zeta[2,1,2]
   - 0.5 zeta[2,2,2] - 0.5 zeta[2,3,2] - 0.5 zeta[2,4,2] - 0.5 zeta[2,5,2]
   - 0.5 zeta[2,6,2] - 0.5 zeta[2,7,2] - 0.5 zeta[8,1,2] - 0.5 zeta[8,2,2]
   - 0.5 zeta[8,3,2] - 0.5 zeta[8,4,2] - 0.5 zeta[8,5,2] - 0.5 zeta[8,6,2]
   - 0.5 zeta[8,7,2] <= 0.02
 R123: 0.5 zeta[9,1,2] + 0.5 zeta[9,2,2] + 0.5 zeta[9,3,2]
   + 0.5 zeta[9,4,2] + 0.5 zeta[9,5,2] + 0.5 zeta[9,6,2] + 0.5 zeta[9,7,2]
   + 0.5 zeta[4,1,2] + 0.5 zeta[4,2,2] + 0.5 zeta[4,3,2] + 0.5 zeta[4,4,2]
   + 0.5 zeta[4,5,2] + 0.5 zeta[4,6,2] + 0.5 zeta[4,7,2] - 0.5 zeta[2,1,2]
   - 0.5 zeta[2,2,2] - 0.5 zeta[2,3,2] - 0.5 zeta[2,4,2] - 0.5 zeta[2,5,2]
   - 0.5 zeta[2,6,2] - 0.5 zeta[2,7,2] - 0.5 zeta[8,1,2] - 0.5 zeta[8,2,2]
   - 0.5 zeta[8,3,2] - 0.5 zeta[8,4,2] - 0.5 zeta[8,5,2] - 0.5 zeta[8,6,2]
   - 0.5 zeta[8,7,2] >= -0.02
Bounds
 Constant = 1
Binaries
 b[1,priors_count.1] b[1,priors_count.2] b[1,priors_count.3]
 b[1,priors_count.4] b[1,priors_count.5] b[1,priors_count.6]
 b[1,priors_count.7] b[1,priors_count.8] b[1,two_year_recid.0]
 b[1,gender_factor.1] b[1,age_factor.1] b[1,age_factor.2] b[1,age_factor.3]
 b[1,race_factor.1] b[1,race_factor.2] b[1,race_factor.3]
 b[1,race_factor.4] b[1,race_factor.5] b[1,race_factor.6]
 b[1,crime_factor.1] b[2,priors_count.1] b[2,priors_count.2]
 b[2,priors_count.3] b[2,priors_count.4] b[2,priors_count.5]
 b[2,priors_count.6] b[2,priors_count.7] b[2,priors_count.8]
 b[2,two_year_recid.0] b[2,gender_factor.1] b[2,age_factor.1]
 b[2,age_factor.2] b[2,age_factor.3] b[2,race_factor.1] b[2,race_factor.2]
 b[2,race_factor.3] b[2,race_factor.4] b[2,race_factor.5]
 b[2,race_factor.6] b[2,crime_factor.1] b[3,priors_count.1]
 b[3,priors_count.2] b[3,priors_count.3] b[3,priors_count.4]
 b[3,priors_count.5] b[3,priors_count.6] b[3,priors_count.7]
 b[3,priors_count.8] b[3,two_year_recid.0] b[3,gender_factor.1]
 b[3,age_factor.1] b[3,age_factor.2] b[3,age_factor.3] b[3,race_factor.1]
 b[3,race_factor.2] b[3,race_factor.3] b[3,race_factor.4]
 b[3,race_factor.5] b[3,race_factor.6] b[3,crime_factor.1] p[1] p[2] p[3]
 p[4] p[5] p[6] p[7]
End
