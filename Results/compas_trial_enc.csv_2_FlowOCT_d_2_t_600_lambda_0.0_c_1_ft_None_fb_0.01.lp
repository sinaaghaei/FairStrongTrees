\ Model FlowOCT
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  zeta[22,1,1] + zeta[22,2,1] + zeta[22,3,1] + zeta[22,4,1] + zeta[22,5,1]
   + zeta[22,6,1] + zeta[22,7,1] + zeta[9,1,1] + zeta[9,2,1] + zeta[9,3,1]
   + zeta[9,4,1] + zeta[9,5,1] + zeta[9,6,1] + zeta[9,7,1] + zeta[5,1,1]
   + zeta[5,2,1] + zeta[5,3,1] + zeta[5,4,1] + zeta[5,5,1] + zeta[5,6,1]
   + zeta[5,7,1] + zeta[8,1,2] + zeta[8,2,2] + zeta[8,3,2] + zeta[8,4,2]
   + zeta[8,5,2] + zeta[8,6,2] + zeta[8,7,2] + zeta[24,1,1] + zeta[24,2,1]
   + zeta[24,3,1] + zeta[24,4,1] + zeta[24,5,1] + zeta[24,6,1]
   + zeta[24,7,1] + zeta[12,1,1] + zeta[12,2,1] + zeta[12,3,1]
   + zeta[12,4,1] + zeta[12,5,1] + zeta[12,6,1] + zeta[12,7,1]
   + zeta[13,1,1] + zeta[13,2,1] + zeta[13,3,1] + zeta[13,4,1]
   + zeta[13,5,1] + zeta[13,6,1] + zeta[13,7,1] + zeta[6,1,1] + zeta[6,2,1]
   + zeta[6,3,1] + zeta[6,4,1] + zeta[6,5,1] + zeta[6,6,1] + zeta[6,7,1]
   + zeta[19,1,1] + zeta[19,2,1] + zeta[19,3,1] + zeta[19,4,1]
   + zeta[19,5,1] + zeta[19,6,1] + zeta[19,7,1] + zeta[23,1,1]
   + zeta[23,2,1] + zeta[23,3,1] + zeta[23,4,1] + zeta[23,5,1]
   + zeta[23,6,1] + zeta[23,7,1] + zeta[11,1,1] + zeta[11,2,1]
   + zeta[11,3,1] + zeta[11,4,1] + zeta[11,5,1] + zeta[11,6,1]
   + zeta[11,7,1] + zeta[18,1,2] + zeta[18,2,2] + zeta[18,3,2]
   + zeta[18,4,2] + zeta[18,5,2] + zeta[18,6,2] + zeta[18,7,2]
   - 84 Constant
Subject To
 R0: - zeta[22,1,1] - zeta[22,1,2] + z[22,1] - z[22,2] - z[22,3] = 0
 R1: - zeta[9,1,1] - zeta[9,1,2] + z[9,1] - z[9,2] - z[9,3] = 0
 R2: - zeta[5,1,1] - zeta[5,1,2] + z[5,1] - z[5,2] - z[5,3] = 0
 R3: - zeta[8,1,1] - zeta[8,1,2] + z[8,1] - z[8,2] - z[8,3] = 0
 R4: - zeta[24,1,1] - zeta[24,1,2] + z[24,1] - z[24,2] - z[24,3] = 0
 R5: - zeta[12,1,1] - zeta[12,1,2] + z[12,1] - z[12,2] - z[12,3] = 0
 R6: - zeta[13,1,1] - zeta[13,1,2] + z[13,1] - z[13,2] - z[13,3] = 0
 R7: - zeta[6,1,1] - zeta[6,1,2] + z[6,1] - z[6,2] - z[6,3] = 0
 R8: - zeta[19,1,1] - zeta[19,1,2] + z[19,1] - z[19,2] - z[19,3] = 0
 R9: - zeta[23,1,1] - zeta[23,1,2] + z[23,1] - z[23,2] - z[23,3] = 0
 R10: - zeta[11,1,1] - zeta[11,1,2] + z[11,1] - z[11,2] - z[11,3] = 0
 R11: - zeta[18,1,1] - zeta[18,1,2] + z[18,1] - z[18,2] - z[18,3] = 0
 R12: - zeta[22,2,1] - zeta[22,2,2] + z[22,2] - z[22,4] - z[22,5] = 0
 R13: - zeta[9,2,1] - zeta[9,2,2] + z[9,2] - z[9,4] - z[9,5] = 0
 R14: - zeta[5,2,1] - zeta[5,2,2] + z[5,2] - z[5,4] - z[5,5] = 0
 R15: - zeta[8,2,1] - zeta[8,2,2] + z[8,2] - z[8,4] - z[8,5] = 0
 R16: - zeta[24,2,1] - zeta[24,2,2] + z[24,2] - z[24,4] - z[24,5] = 0
 R17: - zeta[12,2,1] - zeta[12,2,2] + z[12,2] - z[12,4] - z[12,5] = 0
 R18: - zeta[13,2,1] - zeta[13,2,2] + z[13,2] - z[13,4] - z[13,5] = 0
 R19: - zeta[6,2,1] - zeta[6,2,2] + z[6,2] - z[6,4] - z[6,5] = 0
 R20: - zeta[19,2,1] - zeta[19,2,2] + z[19,2] - z[19,4] - z[19,5] = 0
 R21: - zeta[23,2,1] - zeta[23,2,2] + z[23,2] - z[23,4] - z[23,5] = 0
 R22: - zeta[11,2,1] - zeta[11,2,2] + z[11,2] - z[11,4] - z[11,5] = 0
 R23: - zeta[18,2,1] - zeta[18,2,2] + z[18,2] - z[18,4] - z[18,5] = 0
 R24: - zeta[22,3,1] - zeta[22,3,2] + z[22,3] - z[22,6] - z[22,7] = 0
 R25: - zeta[9,3,1] - zeta[9,3,2] + z[9,3] - z[9,6] - z[9,7] = 0
 R26: - zeta[5,3,1] - zeta[5,3,2] + z[5,3] - z[5,6] - z[5,7] = 0
 R27: - zeta[8,3,1] - zeta[8,3,2] + z[8,3] - z[8,6] - z[8,7] = 0
 R28: - zeta[24,3,1] - zeta[24,3,2] + z[24,3] - z[24,6] - z[24,7] = 0
 R29: - zeta[12,3,1] - zeta[12,3,2] + z[12,3] - z[12,6] - z[12,7] = 0
 R30: - zeta[13,3,1] - zeta[13,3,2] + z[13,3] - z[13,6] - z[13,7] = 0
 R31: - zeta[6,3,1] - zeta[6,3,2] + z[6,3] - z[6,6] - z[6,7] = 0
 R32: - zeta[19,3,1] - zeta[19,3,2] + z[19,3] - z[19,6] - z[19,7] = 0
 R33: - zeta[23,3,1] - zeta[23,3,2] + z[23,3] - z[23,6] - z[23,7] = 0
 R34: - zeta[11,3,1] - zeta[11,3,2] + z[11,3] - z[11,6] - z[11,7] = 0
 R35: - zeta[18,3,1] - zeta[18,3,2] + z[18,3] - z[18,6] - z[18,7] = 0
 R36: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.2]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] - b[1,crime_factor.1] + z[22,2] <= 0
 R37: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.2]
   - b[2,race_factor.3] - b[2,race_factor.4] - b[2,race_factor.5]
   - b[2,race_factor.6] - b[2,crime_factor.1] + z[22,4] <= 0
 R38: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.2]
   - b[3,race_factor.3] - b[3,race_factor.4] - b[3,race_factor.5]
   - b[3,race_factor.6] - b[3,crime_factor.1] + z[22,6] <= 0
 R39: - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[9,2]
   <= 0
 R40: - b[2,gender_factor.1] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1] + z[9,4]
   <= 0
 R41: - b[3,gender_factor.1] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1] + z[9,6]
   <= 0
 R42: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] + z[5,2] <= 0
 R43: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] + z[5,4] <= 0
 R44: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] + z[5,6] <= 0
 R45: - b[1,two_year_recid.0] - b[1,age_factor.1] - b[1,age_factor.2]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[8,2]
   <= 0
 R46: - b[2,two_year_recid.0] - b[2,age_factor.1] - b[2,age_factor.2]
   - b[2,race_factor.1] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1] + z[8,4]
   <= 0
 R47: - b[3,two_year_recid.0] - b[3,age_factor.1] - b[3,age_factor.2]
   - b[3,race_factor.1] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1] + z[8,6]
   <= 0
 R48: - b[1,priors_count.1] - b[1,two_year_recid.0] - b[1,age_factor.2]
   - b[1,age_factor.3] - b[1,race_factor.1] - b[1,race_factor.3]
   - b[1,race_factor.4] - b[1,race_factor.5] - b[1,race_factor.6] + z[24,2]
   <= 0
 R49: - b[2,priors_count.1] - b[2,two_year_recid.0] - b[2,age_factor.2]
   - b[2,age_factor.3] - b[2,race_factor.1] - b[2,race_factor.3]
   - b[2,race_factor.4] - b[2,race_factor.5] - b[2,race_factor.6] + z[24,4]
   <= 0
 R50: - b[3,priors_count.1] - b[3,two_year_recid.0] - b[3,age_factor.2]
   - b[3,age_factor.3] - b[3,race_factor.1] - b[3,race_factor.3]
   - b[3,race_factor.4] - b[3,race_factor.5] - b[3,race_factor.6] + z[24,6]
   <= 0
 R51: - b[1,priors_count.1] - b[1,two_year_recid.0] - b[1,age_factor.2]
   - b[1,age_factor.3] - b[1,race_factor.1] - b[1,race_factor.3]
   - b[1,race_factor.4] - b[1,race_factor.5] - b[1,race_factor.6] + z[12,2]
   <= 0
 R52: - b[2,priors_count.1] - b[2,two_year_recid.0] - b[2,age_factor.2]
   - b[2,age_factor.3] - b[2,race_factor.1] - b[2,race_factor.3]
   - b[2,race_factor.4] - b[2,race_factor.5] - b[2,race_factor.6] + z[12,4]
   <= 0
 R53: - b[3,priors_count.1] - b[3,two_year_recid.0] - b[3,age_factor.2]
   - b[3,age_factor.3] - b[3,race_factor.1] - b[3,race_factor.3]
   - b[3,race_factor.4] - b[3,race_factor.5] - b[3,race_factor.6] + z[12,6]
   <= 0
 R54: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.5]
   - b[1,race_factor.6] - b[1,crime_factor.1] + z[13,2] <= 0
 R55: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.5]
   - b[2,race_factor.6] - b[2,crime_factor.1] + z[13,4] <= 0
 R56: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.5]
   - b[3,race_factor.6] - b[3,crime_factor.1] + z[13,6] <= 0
 R57: - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[6,2]
   <= 0
 R58: - b[2,gender_factor.1] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1] + z[6,4]
   <= 0
 R59: - b[3,gender_factor.1] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1] + z[6,6]
   <= 0
 R60: - b[1,two_year_recid.0] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1]
   + z[19,2] <= 0
 R61: - b[2,two_year_recid.0] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1]
   + z[19,4] <= 0
 R62: - b[3,two_year_recid.0] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1]
   + z[19,6] <= 0
 R63: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,crime_factor.1] + z[23,2] <= 0
 R64: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,crime_factor.1] + z[23,4] <= 0
 R65: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,crime_factor.1] + z[23,6] <= 0
 R66: - b[1,two_year_recid.0] - b[1,gender_factor.1] - b[1,age_factor.1]
   - b[1,age_factor.3] - b[1,race_factor.2] - b[1,race_factor.3]
   - b[1,race_factor.4] - b[1,race_factor.5] - b[1,race_factor.6] + z[11,2]
   <= 0
 R67: - b[2,two_year_recid.0] - b[2,gender_factor.1] - b[2,age_factor.1]
   - b[2,age_factor.3] - b[2,race_factor.2] - b[2,race_factor.3]
   - b[2,race_factor.4] - b[2,race_factor.5] - b[2,race_factor.6] + z[11,4]
   <= 0
 R68: - b[3,two_year_recid.0] - b[3,gender_factor.1] - b[3,age_factor.1]
   - b[3,age_factor.3] - b[3,race_factor.2] - b[3,race_factor.3]
   - b[3,race_factor.4] - b[3,race_factor.5] - b[3,race_factor.6] + z[11,6]
   <= 0
 R69: - b[1,two_year_recid.0] - b[1,age_factor.1] - b[1,age_factor.2]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] + z[18,2] <= 0
 R70: - b[2,two_year_recid.0] - b[2,age_factor.1] - b[2,age_factor.2]
   - b[2,race_factor.1] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] + z[18,4] <= 0
 R71: - b[3,two_year_recid.0] - b[3,age_factor.1] - b[3,age_factor.2]
   - b[3,race_factor.1] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] + z[18,6] <= 0
 R72: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.1]
   + z[22,3] <= 0
 R73: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.1]
   + z[22,5] <= 0
 R74: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.1]
   + z[22,7] <= 0
 R75: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,age_factor.1] - b[1,race_factor.1] + z[9,3] <= 0
 R76: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,age_factor.1] - b[2,race_factor.1] + z[9,5] <= 0
 R77: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,age_factor.1] - b[3,race_factor.1] + z[9,7] <= 0
 R78: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.6]
   - b[1,crime_factor.1] + z[5,3] <= 0
 R79: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.6]
   - b[2,crime_factor.1] + z[5,5] <= 0
 R80: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.6]
   - b[3,crime_factor.1] + z[5,7] <= 0
 R81: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.3] - b[1,race_factor.2] + z[8,3] <= 0
 R82: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.3] - b[2,race_factor.2] + z[8,5] <= 0
 R83: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.3] - b[3,race_factor.2] + z[8,7] <= 0
 R84: - b[1,priors_count.2] - b[1,priors_count.3] - b[1,priors_count.4]
   - b[1,priors_count.5] - b[1,priors_count.6] - b[1,priors_count.7]
   - b[1,priors_count.8] - b[1,gender_factor.1] - b[1,age_factor.1]
   - b[1,race_factor.2] - b[1,crime_factor.1] + z[24,3] <= 0
 R85: - b[2,priors_count.2] - b[2,priors_count.3] - b[2,priors_count.4]
   - b[2,priors_count.5] - b[2,priors_count.6] - b[2,priors_count.7]
   - b[2,priors_count.8] - b[2,gender_factor.1] - b[2,age_factor.1]
   - b[2,race_factor.2] - b[2,crime_factor.1] + z[24,5] <= 0
 R86: - b[3,priors_count.2] - b[3,priors_count.3] - b[3,priors_count.4]
   - b[3,priors_count.5] - b[3,priors_count.6] - b[3,priors_count.7]
   - b[3,priors_count.8] - b[3,gender_factor.1] - b[3,age_factor.1]
   - b[3,race_factor.2] - b[3,crime_factor.1] + z[24,7] <= 0
 R87: - b[1,priors_count.2] - b[1,priors_count.3] - b[1,priors_count.4]
   - b[1,priors_count.5] - b[1,priors_count.6] - b[1,priors_count.7]
   - b[1,priors_count.8] - b[1,gender_factor.1] - b[1,age_factor.1]
   - b[1,race_factor.2] - b[1,crime_factor.1] + z[12,3] <= 0
 R88: - b[2,priors_count.2] - b[2,priors_count.3] - b[2,priors_count.4]
   - b[2,priors_count.5] - b[2,priors_count.6] - b[2,priors_count.7]
   - b[2,priors_count.8] - b[2,gender_factor.1] - b[2,age_factor.1]
   - b[2,race_factor.2] - b[2,crime_factor.1] + z[12,5] <= 0
 R89: - b[3,priors_count.2] - b[3,priors_count.3] - b[3,priors_count.4]
   - b[3,priors_count.5] - b[3,priors_count.6] - b[3,priors_count.7]
   - b[3,priors_count.8] - b[3,gender_factor.1] - b[3,age_factor.1]
   - b[3,race_factor.2] - b[3,crime_factor.1] + z[12,7] <= 0
 R90: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.4]
   + z[13,3] <= 0
 R91: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.4]
   + z[13,5] <= 0
 R92: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.4]
   + z[13,7] <= 0
 R93: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,age_factor.1] - b[1,race_factor.1] + z[6,3] <= 0
 R94: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,age_factor.1] - b[2,race_factor.1] + z[6,5] <= 0
 R95: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,age_factor.1] - b[3,race_factor.1] + z[6,7] <= 0
 R96: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.1] - b[1,race_factor.1] + z[19,3] <= 0
 R97: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.1] - b[2,race_factor.1] + z[19,5] <= 0
 R98: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.1] - b[3,race_factor.1] + z[19,7] <= 0
 R99: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.6]
   + z[23,3] <= 0
 R100: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.6]
   + z[23,5] <= 0
 R101: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.6]
   + z[23,7] <= 0
 R102: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,age_factor.2]
   - b[1,race_factor.1] - b[1,crime_factor.1] + z[11,3] <= 0
 R103: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,age_factor.2]
   - b[2,race_factor.1] - b[2,crime_factor.1] + z[11,5] <= 0
 R104: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,age_factor.2]
   - b[3,race_factor.1] - b[3,crime_factor.1] + z[11,7] <= 0
 R105: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.3] - b[1,race_factor.2] - b[1,crime_factor.1] + z[18,3]
   <= 0
 R106: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.3] - b[2,race_factor.2] - b[2,crime_factor.1] + z[18,5]
   <= 0
 R107: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.3] - b[3,race_factor.2] - b[3,crime_factor.1] + z[18,7]
   <= 0
 R108: b[1,priors_count.1] + b[1,priors_count.2] + b[1,priors_count.3]
   + b[1,priors_count.4] + b[1,priors_count.5] + b[1,priors_count.6]
   + b[1,priors_count.7] + b[1,priors_count.8] + b[1,two_year_recid.0]
   + b[1,gender_factor.1] + b[1,age_factor.1] + b[1,age_factor.2]
   + b[1,age_factor.3] + b[1,race_factor.1] + b[1,race_factor.2]
   + b[1,race_factor.3] + b[1,race_factor.4] + b[1,race_factor.5]
   + b[1,race_factor.6] + b[1,crime_factor.1] + p[1] = 1
 R109: b[2,priors_count.1] + b[2,priors_count.2] + b[2,priors_count.3]
   + b[2,priors_count.4] + b[2,priors_count.5] + b[2,priors_count.6]
   + b[2,priors_count.7] + b[2,priors_count.8] + b[2,two_year_recid.0]
   + b[2,gender_factor.1] + b[2,age_factor.1] + b[2,age_factor.2]
   + b[2,age_factor.3] + b[2,race_factor.1] + b[2,race_factor.2]
   + b[2,race_factor.3] + b[2,race_factor.4] + b[2,race_factor.5]
   + b[2,race_factor.6] + b[2,crime_factor.1] + p[1] + p[2] = 1
 R110: b[3,priors_count.1] + b[3,priors_count.2] + b[3,priors_count.3]
   + b[3,priors_count.4] + b[3,priors_count.5] + b[3,priors_count.6]
   + b[3,priors_count.7] + b[3,priors_count.8] + b[3,two_year_recid.0]
   + b[3,gender_factor.1] + b[3,age_factor.1] + b[3,age_factor.2]
   + b[3,age_factor.3] + b[3,race_factor.1] + b[3,race_factor.2]
   + b[3,race_factor.3] + b[3,race_factor.4] + b[3,race_factor.5]
   + b[3,race_factor.6] + b[3,crime_factor.1] + p[1] + p[3] = 1
 R111: p[1] + p[2] + p[4] = 1
 R112: p[1] + p[2] + p[5] = 1
 R113: p[1] + p[3] + p[6] = 1
 R114: p[1] + p[3] + p[7] = 1
 R115: - beta[1,1] + zeta[22,1,1] <= 0
 R116: - beta[1,2] + zeta[22,1,2] <= 0
 R117: - beta[1,1] + zeta[9,1,1] <= 0
 R118: - beta[1,2] + zeta[9,1,2] <= 0
 R119: - beta[1,1] + zeta[5,1,1] <= 0
 R120: - beta[1,2] + zeta[5,1,2] <= 0
 R121: - beta[1,1] + zeta[8,1,1] <= 0
 R122: - beta[1,2] + zeta[8,1,2] <= 0
 R123: - beta[1,1] + zeta[24,1,1] <= 0
 R124: - beta[1,2] + zeta[24,1,2] <= 0
 R125: - beta[1,1] + zeta[12,1,1] <= 0
 R126: - beta[1,2] + zeta[12,1,2] <= 0
 R127: - beta[1,1] + zeta[13,1,1] <= 0
 R128: - beta[1,2] + zeta[13,1,2] <= 0
 R129: - beta[1,1] + zeta[6,1,1] <= 0
 R130: - beta[1,2] + zeta[6,1,2] <= 0
 R131: - beta[1,1] + zeta[19,1,1] <= 0
 R132: - beta[1,2] + zeta[19,1,2] <= 0
 R133: - beta[1,1] + zeta[23,1,1] <= 0
 R134: - beta[1,2] + zeta[23,1,2] <= 0
 R135: - beta[1,1] + zeta[11,1,1] <= 0
 R136: - beta[1,2] + zeta[11,1,2] <= 0
 R137: - beta[1,1] + zeta[18,1,1] <= 0
 R138: - beta[1,2] + zeta[18,1,2] <= 0
 R139: - beta[2,1] + zeta[22,2,1] <= 0
 R140: - beta[2,2] + zeta[22,2,2] <= 0
 R141: - beta[2,1] + zeta[9,2,1] <= 0
 R142: - beta[2,2] + zeta[9,2,2] <= 0
 R143: - beta[2,1] + zeta[5,2,1] <= 0
 R144: - beta[2,2] + zeta[5,2,2] <= 0
 R145: - beta[2,1] + zeta[8,2,1] <= 0
 R146: - beta[2,2] + zeta[8,2,2] <= 0
 R147: - beta[2,1] + zeta[24,2,1] <= 0
 R148: - beta[2,2] + zeta[24,2,2] <= 0
 R149: - beta[2,1] + zeta[12,2,1] <= 0
 R150: - beta[2,2] + zeta[12,2,2] <= 0
 R151: - beta[2,1] + zeta[13,2,1] <= 0
 R152: - beta[2,2] + zeta[13,2,2] <= 0
 R153: - beta[2,1] + zeta[6,2,1] <= 0
 R154: - beta[2,2] + zeta[6,2,2] <= 0
 R155: - beta[2,1] + zeta[19,2,1] <= 0
 R156: - beta[2,2] + zeta[19,2,2] <= 0
 R157: - beta[2,1] + zeta[23,2,1] <= 0
 R158: - beta[2,2] + zeta[23,2,2] <= 0
 R159: - beta[2,1] + zeta[11,2,1] <= 0
 R160: - beta[2,2] + zeta[11,2,2] <= 0
 R161: - beta[2,1] + zeta[18,2,1] <= 0
 R162: - beta[2,2] + zeta[18,2,2] <= 0
 R163: - beta[3,1] + zeta[22,3,1] <= 0
 R164: - beta[3,2] + zeta[22,3,2] <= 0
 R165: - beta[3,1] + zeta[9,3,1] <= 0
 R166: - beta[3,2] + zeta[9,3,2] <= 0
 R167: - beta[3,1] + zeta[5,3,1] <= 0
 R168: - beta[3,2] + zeta[5,3,2] <= 0
 R169: - beta[3,1] + zeta[8,3,1] <= 0
 R170: - beta[3,2] + zeta[8,3,2] <= 0
 R171: - beta[3,1] + zeta[24,3,1] <= 0
 R172: - beta[3,2] + zeta[24,3,2] <= 0
 R173: - beta[3,1] + zeta[12,3,1] <= 0
 R174: - beta[3,2] + zeta[12,3,2] <= 0
 R175: - beta[3,1] + zeta[13,3,1] <= 0
 R176: - beta[3,2] + zeta[13,3,2] <= 0
 R177: - beta[3,1] + zeta[6,3,1] <= 0
 R178: - beta[3,2] + zeta[6,3,2] <= 0
 R179: - beta[3,1] + zeta[19,3,1] <= 0
 R180: - beta[3,2] + zeta[19,3,2] <= 0
 R181: - beta[3,1] + zeta[23,3,1] <= 0
 R182: - beta[3,2] + zeta[23,3,2] <= 0
 R183: - beta[3,1] + zeta[11,3,1] <= 0
 R184: - beta[3,2] + zeta[11,3,2] <= 0
 R185: - beta[3,1] + zeta[18,3,1] <= 0
 R186: - beta[3,2] + zeta[18,3,2] <= 0
 R187: - beta[4,1] + zeta[22,4,1] <= 0
 R188: - beta[4,2] + zeta[22,4,2] <= 0
 R189: - beta[4,1] + zeta[9,4,1] <= 0
 R190: - beta[4,2] + zeta[9,4,2] <= 0
 R191: - beta[4,1] + zeta[5,4,1] <= 0
 R192: - beta[4,2] + zeta[5,4,2] <= 0
 R193: - beta[4,1] + zeta[8,4,1] <= 0
 R194: - beta[4,2] + zeta[8,4,2] <= 0
 R195: - beta[4,1] + zeta[24,4,1] <= 0
 R196: - beta[4,2] + zeta[24,4,2] <= 0
 R197: - beta[4,1] + zeta[12,4,1] <= 0
 R198: - beta[4,2] + zeta[12,4,2] <= 0
 R199: - beta[4,1] + zeta[13,4,1] <= 0
 R200: - beta[4,2] + zeta[13,4,2] <= 0
 R201: - beta[4,1] + zeta[6,4,1] <= 0
 R202: - beta[4,2] + zeta[6,4,2] <= 0
 R203: - beta[4,1] + zeta[19,4,1] <= 0
 R204: - beta[4,2] + zeta[19,4,2] <= 0
 R205: - beta[4,1] + zeta[23,4,1] <= 0
 R206: - beta[4,2] + zeta[23,4,2] <= 0
 R207: - beta[4,1] + zeta[11,4,1] <= 0
 R208: - beta[4,2] + zeta[11,4,2] <= 0
 R209: - beta[4,1] + zeta[18,4,1] <= 0
 R210: - beta[4,2] + zeta[18,4,2] <= 0
 R211: - beta[5,1] + zeta[22,5,1] <= 0
 R212: - beta[5,2] + zeta[22,5,2] <= 0
 R213: - beta[5,1] + zeta[9,5,1] <= 0
 R214: - beta[5,2] + zeta[9,5,2] <= 0
 R215: - beta[5,1] + zeta[5,5,1] <= 0
 R216: - beta[5,2] + zeta[5,5,2] <= 0
 R217: - beta[5,1] + zeta[8,5,1] <= 0
 R218: - beta[5,2] + zeta[8,5,2] <= 0
 R219: - beta[5,1] + zeta[24,5,1] <= 0
 R220: - beta[5,2] + zeta[24,5,2] <= 0
 R221: - beta[5,1] + zeta[12,5,1] <= 0
 R222: - beta[5,2] + zeta[12,5,2] <= 0
 R223: - beta[5,1] + zeta[13,5,1] <= 0
 R224: - beta[5,2] + zeta[13,5,2] <= 0
 R225: - beta[5,1] + zeta[6,5,1] <= 0
 R226: - beta[5,2] + zeta[6,5,2] <= 0
 R227: - beta[5,1] + zeta[19,5,1] <= 0
 R228: - beta[5,2] + zeta[19,5,2] <= 0
 R229: - beta[5,1] + zeta[23,5,1] <= 0
 R230: - beta[5,2] + zeta[23,5,2] <= 0
 R231: - beta[5,1] + zeta[11,5,1] <= 0
 R232: - beta[5,2] + zeta[11,5,2] <= 0
 R233: - beta[5,1] + zeta[18,5,1] <= 0
 R234: - beta[5,2] + zeta[18,5,2] <= 0
 R235: - beta[6,1] + zeta[22,6,1] <= 0
 R236: - beta[6,2] + zeta[22,6,2] <= 0
 R237: - beta[6,1] + zeta[9,6,1] <= 0
 R238: - beta[6,2] + zeta[9,6,2] <= 0
 R239: - beta[6,1] + zeta[5,6,1] <= 0
 R240: - beta[6,2] + zeta[5,6,2] <= 0
 R241: - beta[6,1] + zeta[8,6,1] <= 0
 R242: - beta[6,2] + zeta[8,6,2] <= 0
 R243: - beta[6,1] + zeta[24,6,1] <= 0
 R244: - beta[6,2] + zeta[24,6,2] <= 0
 R245: - beta[6,1] + zeta[12,6,1] <= 0
 R246: - beta[6,2] + zeta[12,6,2] <= 0
 R247: - beta[6,1] + zeta[13,6,1] <= 0
 R248: - beta[6,2] + zeta[13,6,2] <= 0
 R249: - beta[6,1] + zeta[6,6,1] <= 0
 R250: - beta[6,2] + zeta[6,6,2] <= 0
 R251: - beta[6,1] + zeta[19,6,1] <= 0
 R252: - beta[6,2] + zeta[19,6,2] <= 0
 R253: - beta[6,1] + zeta[23,6,1] <= 0
 R254: - beta[6,2] + zeta[23,6,2] <= 0
 R255: - beta[6,1] + zeta[11,6,1] <= 0
 R256: - beta[6,2] + zeta[11,6,2] <= 0
 R257: - beta[6,1] + zeta[18,6,1] <= 0
 R258: - beta[6,2] + zeta[18,6,2] <= 0
 R259: - beta[7,1] + zeta[22,7,1] <= 0
 R260: - beta[7,2] + zeta[22,7,2] <= 0
 R261: - beta[7,1] + zeta[9,7,1] <= 0
 R262: - beta[7,2] + zeta[9,7,2] <= 0
 R263: - beta[7,1] + zeta[5,7,1] <= 0
 R264: - beta[7,2] + zeta[5,7,2] <= 0
 R265: - beta[7,1] + zeta[8,7,1] <= 0
 R266: - beta[7,2] + zeta[8,7,2] <= 0
 R267: - beta[7,1] + zeta[24,7,1] <= 0
 R268: - beta[7,2] + zeta[24,7,2] <= 0
 R269: - beta[7,1] + zeta[12,7,1] <= 0
 R270: - beta[7,2] + zeta[12,7,2] <= 0
 R271: - beta[7,1] + zeta[13,7,1] <= 0
 R272: - beta[7,2] + zeta[13,7,2] <= 0
 R273: - beta[7,1] + zeta[6,7,1] <= 0
 R274: - beta[7,2] + zeta[6,7,2] <= 0
 R275: - beta[7,1] + zeta[19,7,1] <= 0
 R276: - beta[7,2] + zeta[19,7,2] <= 0
 R277: - beta[7,1] + zeta[23,7,1] <= 0
 R278: - beta[7,2] + zeta[23,7,2] <= 0
 R279: - beta[7,1] + zeta[11,7,1] <= 0
 R280: - beta[7,2] + zeta[11,7,2] <= 0
 R281: - beta[7,1] + zeta[18,7,1] <= 0
 R282: - beta[7,2] + zeta[18,7,2] <= 0
 R283: - p[1] + beta[1,1] + beta[1,2] = 0
 R284: - p[2] + beta[2,1] + beta[2,2] = 0
 R285: - p[3] + beta[3,1] + beta[3,2] = 0
 R286: - p[4] + beta[4,1] + beta[4,2] = 0
 R287: - p[5] + beta[5,1] + beta[5,2] = 0
 R288: - p[6] + beta[6,1] + beta[6,2] = 0
 R289: - p[7] + beta[7,1] + beta[7,2] = 0
 R290: zeta[22,4,1] + zeta[22,4,2] - z[22,4] = 0
 R291: zeta[9,4,1] + zeta[9,4,2] - z[9,4] = 0
 R292: zeta[5,4,1] + zeta[5,4,2] - z[5,4] = 0
 R293: zeta[8,4,1] + zeta[8,4,2] - z[8,4] = 0
 R294: zeta[24,4,1] + zeta[24,4,2] - z[24,4] = 0
 R295: zeta[12,4,1] + zeta[12,4,2] - z[12,4] = 0
 R296: zeta[13,4,1] + zeta[13,4,2] - z[13,4] = 0
 R297: zeta[6,4,1] + zeta[6,4,2] - z[6,4] = 0
 R298: zeta[19,4,1] + zeta[19,4,2] - z[19,4] = 0
 R299: zeta[23,4,1] + zeta[23,4,2] - z[23,4] = 0
 R300: zeta[11,4,1] + zeta[11,4,2] - z[11,4] = 0
 R301: zeta[18,4,1] + zeta[18,4,2] - z[18,4] = 0
 R302: zeta[22,5,1] + zeta[22,5,2] - z[22,5] = 0
 R303: zeta[9,5,1] + zeta[9,5,2] - z[9,5] = 0
 R304: zeta[5,5,1] + zeta[5,5,2] - z[5,5] = 0
 R305: zeta[8,5,1] + zeta[8,5,2] - z[8,5] = 0
 R306: zeta[24,5,1] + zeta[24,5,2] - z[24,5] = 0
 R307: zeta[12,5,1] + zeta[12,5,2] - z[12,5] = 0
 R308: zeta[13,5,1] + zeta[13,5,2] - z[13,5] = 0
 R309: zeta[6,5,1] + zeta[6,5,2] - z[6,5] = 0
 R310: zeta[19,5,1] + zeta[19,5,2] - z[19,5] = 0
 R311: zeta[23,5,1] + zeta[23,5,2] - z[23,5] = 0
 R312: zeta[11,5,1] + zeta[11,5,2] - z[11,5] = 0
 R313: zeta[18,5,1] + zeta[18,5,2] - z[18,5] = 0
 R314: zeta[22,6,1] + zeta[22,6,2] - z[22,6] = 0
 R315: zeta[9,6,1] + zeta[9,6,2] - z[9,6] = 0
 R316: zeta[5,6,1] + zeta[5,6,2] - z[5,6] = 0
 R317: zeta[8,6,1] + zeta[8,6,2] - z[8,6] = 0
 R318: zeta[24,6,1] + zeta[24,6,2] - z[24,6] = 0
 R319: zeta[12,6,1] + zeta[12,6,2] - z[12,6] = 0
 R320: zeta[13,6,1] + zeta[13,6,2] - z[13,6] = 0
 R321: zeta[6,6,1] + zeta[6,6,2] - z[6,6] = 0
 R322: zeta[19,6,1] + zeta[19,6,2] - z[19,6] = 0
 R323: zeta[23,6,1] + zeta[23,6,2] - z[23,6] = 0
 R324: zeta[11,6,1] + zeta[11,6,2] - z[11,6] = 0
 R325: zeta[18,6,1] + zeta[18,6,2] - z[18,6] = 0
 R326: zeta[22,7,1] + zeta[22,7,2] - z[22,7] = 0
 R327: zeta[9,7,1] + zeta[9,7,2] - z[9,7] = 0
 R328: zeta[5,7,1] + zeta[5,7,2] - z[5,7] = 0
 R329: zeta[8,7,1] + zeta[8,7,2] - z[8,7] = 0
 R330: zeta[24,7,1] + zeta[24,7,2] - z[24,7] = 0
 R331: zeta[12,7,1] + zeta[12,7,2] - z[12,7] = 0
 R332: zeta[13,7,1] + zeta[13,7,2] - z[13,7] = 0
 R333: zeta[6,7,1] + zeta[6,7,2] - z[6,7] = 0
 R334: zeta[19,7,1] + zeta[19,7,2] - z[19,7] = 0
 R335: zeta[23,7,1] + zeta[23,7,2] - z[23,7] = 0
 R336: zeta[11,7,1] + zeta[11,7,2] - z[11,7] = 0
 R337: zeta[18,7,1] + zeta[18,7,2] - z[18,7] = 0
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
 p[4] p[5] p[6] p[7] zeta[22,1,1] zeta[22,1,2] zeta[22,2,1] zeta[22,2,2]
 zeta[22,3,1] zeta[22,3,2] zeta[22,4,1] zeta[22,4,2] zeta[22,5,1]
 zeta[22,5,2] zeta[22,6,1] zeta[22,6,2] zeta[22,7,1] zeta[22,7,2]
 zeta[9,1,1] zeta[9,1,2] zeta[9,2,1] zeta[9,2,2] zeta[9,3,1] zeta[9,3,2]
 zeta[9,4,1] zeta[9,4,2] zeta[9,5,1] zeta[9,5,2] zeta[9,6,1] zeta[9,6,2]
 zeta[9,7,1] zeta[9,7,2] zeta[5,1,1] zeta[5,1,2] zeta[5,2,1] zeta[5,2,2]
 zeta[5,3,1] zeta[5,3,2] zeta[5,4,1] zeta[5,4,2] zeta[5,5,1] zeta[5,5,2]
 zeta[5,6,1] zeta[5,6,2] zeta[5,7,1] zeta[5,7,2] zeta[8,1,1] zeta[8,1,2]
 zeta[8,2,1] zeta[8,2,2] zeta[8,3,1] zeta[8,3,2] zeta[8,4,1] zeta[8,4,2]
 zeta[8,5,1] zeta[8,5,2] zeta[8,6,1] zeta[8,6,2] zeta[8,7,1] zeta[8,7,2]
 zeta[24,1,1] zeta[24,1,2] zeta[24,2,1] zeta[24,2,2] zeta[24,3,1]
 zeta[24,3,2] zeta[24,4,1] zeta[24,4,2] zeta[24,5,1] zeta[24,5,2]
 zeta[24,6,1] zeta[24,6,2] zeta[24,7,1] zeta[24,7,2] zeta[12,1,1]
 zeta[12,1,2] zeta[12,2,1] zeta[12,2,2] zeta[12,3,1] zeta[12,3,2]
 zeta[12,4,1] zeta[12,4,2] zeta[12,5,1] zeta[12,5,2] zeta[12,6,1]
 zeta[12,6,2] zeta[12,7,1] zeta[12,7,2] zeta[13,1,1] zeta[13,1,2]
 zeta[13,2,1] zeta[13,2,2] zeta[13,3,1] zeta[13,3,2] zeta[13,4,1]
 zeta[13,4,2] zeta[13,5,1] zeta[13,5,2] zeta[13,6,1] zeta[13,6,2]
 zeta[13,7,1] zeta[13,7,2] zeta[6,1,1] zeta[6,1,2] zeta[6,2,1] zeta[6,2,2]
 zeta[6,3,1] zeta[6,3,2] zeta[6,4,1] zeta[6,4,2] zeta[6,5,1] zeta[6,5,2]
 zeta[6,6,1] zeta[6,6,2] zeta[6,7,1] zeta[6,7,2] zeta[19,1,1] zeta[19,1,2]
 zeta[19,2,1] zeta[19,2,2] zeta[19,3,1] zeta[19,3,2] zeta[19,4,1]
 zeta[19,4,2] zeta[19,5,1] zeta[19,5,2] zeta[19,6,1] zeta[19,6,2]
 zeta[19,7,1] zeta[19,7,2] zeta[23,1,1] zeta[23,1,2] zeta[23,2,1]
 zeta[23,2,2] zeta[23,3,1] zeta[23,3,2] zeta[23,4,1] zeta[23,4,2]
 zeta[23,5,1] zeta[23,5,2] zeta[23,6,1] zeta[23,6,2] zeta[23,7,1]
 zeta[23,7,2] zeta[11,1,1] zeta[11,1,2] zeta[11,2,1] zeta[11,2,2]
 zeta[11,3,1] zeta[11,3,2] zeta[11,4,1] zeta[11,4,2] zeta[11,5,1]
 zeta[11,5,2] zeta[11,6,1] zeta[11,6,2] zeta[11,7,1] zeta[11,7,2]
 zeta[18,1,1] zeta[18,1,2] zeta[18,2,1] zeta[18,2,2] zeta[18,3,1]
 zeta[18,3,2] zeta[18,4,1] zeta[18,4,2] zeta[18,5,1] zeta[18,5,2]
 zeta[18,6,1] zeta[18,6,2] zeta[18,7,1] zeta[18,7,2] z[22,1] z[22,2]
 z[22,3] z[22,4] z[22,5] z[22,6] z[22,7] z[9,1] z[9,2] z[9,3] z[9,4] z[9,5]
 z[9,6] z[9,7] z[5,1] z[5,2] z[5,3] z[5,4] z[5,5] z[5,6] z[5,7] z[8,1]
 z[8,2] z[8,3] z[8,4] z[8,5] z[8,6] z[8,7] z[24,1] z[24,2] z[24,3] z[24,4]
 z[24,5] z[24,6] z[24,7] z[12,1] z[12,2] z[12,3] z[12,4] z[12,5] z[12,6]
 z[12,7] z[13,1] z[13,2] z[13,3] z[13,4] z[13,5] z[13,6] z[13,7] z[6,1]
 z[6,2] z[6,3] z[6,4] z[6,5] z[6,6] z[6,7] z[19,1] z[19,2] z[19,3] z[19,4]
 z[19,5] z[19,6] z[19,7] z[23,1] z[23,2] z[23,3] z[23,4] z[23,5] z[23,6]
 z[23,7] z[11,1] z[11,2] z[11,3] z[11,4] z[11,5] z[11,6] z[11,7] z[18,1]
 z[18,2] z[18,3] z[18,4] z[18,5] z[18,6] z[18,7]
End
