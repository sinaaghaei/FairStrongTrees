\ Model FlowOCT
\ LP format - for model browsing. Use MPS format to capture full model detail.
Maximize
  zeta[43,1,1] + zeta[43,2,1] + zeta[43,3,1] + zeta[43,4,1] + zeta[43,5,1]
   + zeta[43,6,1] + zeta[43,7,1] + zeta[1,1,1] + zeta[1,2,1] + zeta[1,3,1]
   + zeta[1,4,1] + zeta[1,5,1] + zeta[1,6,1] + zeta[1,7,1] + zeta[46,1,2]
   + zeta[46,2,2] + zeta[46,3,2] + zeta[46,4,2] + zeta[46,5,2]
   + zeta[46,6,2] + zeta[46,7,2] + zeta[38,1,1] + zeta[38,2,1]
   + zeta[38,3,1] + zeta[38,4,1] + zeta[38,5,1] + zeta[38,6,1]
   + zeta[38,7,1] + zeta[16,1,1] + zeta[16,2,1] + zeta[16,3,1]
   + zeta[16,4,1] + zeta[16,5,1] + zeta[16,6,1] + zeta[16,7,1]
   + zeta[27,1,2] + zeta[27,2,2] + zeta[27,3,2] + zeta[27,4,2]
   + zeta[27,5,2] + zeta[27,6,2] + zeta[27,7,2] + zeta[3,1,1] + zeta[3,2,1]
   + zeta[3,3,1] + zeta[3,4,1] + zeta[3,5,1] + zeta[3,6,1] + zeta[3,7,1]
   + zeta[24,1,1] + zeta[24,2,1] + zeta[24,3,1] + zeta[24,4,1]
   + zeta[24,5,1] + zeta[24,6,1] + zeta[24,7,1] + zeta[21,1,1]
   + zeta[21,2,1] + zeta[21,3,1] + zeta[21,4,1] + zeta[21,5,1]
   + zeta[21,6,1] + zeta[21,7,1] + zeta[49,1,1] + zeta[49,2,1]
   + zeta[49,3,1] + zeta[49,4,1] + zeta[49,5,1] + zeta[49,6,1]
   + zeta[49,7,1] + zeta[48,1,2] + zeta[48,2,2] + zeta[48,3,2]
   + zeta[48,4,2] + zeta[48,5,2] + zeta[48,6,2] + zeta[48,7,2]
   + zeta[34,1,1] + zeta[34,2,1] + zeta[34,3,1] + zeta[34,4,1]
   + zeta[34,5,1] + zeta[34,6,1] + zeta[34,7,1] + zeta[9,1,1] + zeta[9,2,1]
   + zeta[9,3,1] + zeta[9,4,1] + zeta[9,5,1] + zeta[9,6,1] + zeta[9,7,1]
   + zeta[45,1,1] + zeta[45,2,1] + zeta[45,3,1] + zeta[45,4,1]
   + zeta[45,5,1] + zeta[45,6,1] + zeta[45,7,1] + zeta[23,1,1]
   + zeta[23,2,1] + zeta[23,3,1] + zeta[23,4,1] + zeta[23,5,1]
   + zeta[23,6,1] + zeta[23,7,1] + zeta[19,1,1] + zeta[19,2,1]
   + zeta[19,3,1] + zeta[19,4,1] + zeta[19,5,1] + zeta[19,6,1]
   + zeta[19,7,1] + zeta[6,1,1] + zeta[6,2,1] + zeta[6,3,1] + zeta[6,4,1]
   + zeta[6,5,1] + zeta[6,6,1] + zeta[6,7,1] + zeta[25,1,2] + zeta[25,2,2]
   + zeta[25,3,2] + zeta[25,4,2] + zeta[25,5,2] + zeta[25,6,2]
   + zeta[25,7,2] + zeta[4,1,2] + zeta[4,2,2] + zeta[4,3,2] + zeta[4,4,2]
   + zeta[4,5,2] + zeta[4,6,2] + zeta[4,7,2] + zeta[26,1,2] + zeta[26,2,2]
   + zeta[26,3,2] + zeta[26,4,2] + zeta[26,5,2] + zeta[26,6,2]
   + zeta[26,7,2] + zeta[41,1,2] + zeta[41,2,2] + zeta[41,3,2]
   + zeta[41,4,2] + zeta[41,5,2] + zeta[41,6,2] + zeta[41,7,2]
   + zeta[31,1,2] + zeta[31,2,2] + zeta[31,3,2] + zeta[31,4,2]
   + zeta[31,5,2] + zeta[31,6,2] + zeta[31,7,2] + zeta[7,1,1] + zeta[7,2,1]
   + zeta[7,3,1] + zeta[7,4,1] + zeta[7,5,1] + zeta[7,6,1] + zeta[7,7,1]
   + zeta[0,1,1] + zeta[0,2,1] + zeta[0,3,1] + zeta[0,4,1] + zeta[0,5,1]
   + zeta[0,6,1] + zeta[0,7,1] - 168 Constant
Subject To
 R0: - zeta[43,1,1] - zeta[43,1,2] + z[43,1] - z[43,2] - z[43,3] = 0
 R1: - zeta[1,1,1] - zeta[1,1,2] + z[1,1] - z[1,2] - z[1,3] = 0
 R2: - zeta[46,1,1] - zeta[46,1,2] + z[46,1] - z[46,2] - z[46,3] = 0
 R3: - zeta[38,1,1] - zeta[38,1,2] + z[38,1] - z[38,2] - z[38,3] = 0
 R4: - zeta[16,1,1] - zeta[16,1,2] + z[16,1] - z[16,2] - z[16,3] = 0
 R5: - zeta[27,1,1] - zeta[27,1,2] + z[27,1] - z[27,2] - z[27,3] = 0
 R6: - zeta[3,1,1] - zeta[3,1,2] + z[3,1] - z[3,2] - z[3,3] = 0
 R7: - zeta[24,1,1] - zeta[24,1,2] + z[24,1] - z[24,2] - z[24,3] = 0
 R8: - zeta[21,1,1] - zeta[21,1,2] + z[21,1] - z[21,2] - z[21,3] = 0
 R9: - zeta[49,1,1] - zeta[49,1,2] + z[49,1] - z[49,2] - z[49,3] = 0
 R10: - zeta[48,1,1] - zeta[48,1,2] + z[48,1] - z[48,2] - z[48,3] = 0
 R11: - zeta[34,1,1] - zeta[34,1,2] + z[34,1] - z[34,2] - z[34,3] = 0
 R12: - zeta[9,1,1] - zeta[9,1,2] + z[9,1] - z[9,2] - z[9,3] = 0
 R13: - zeta[45,1,1] - zeta[45,1,2] + z[45,1] - z[45,2] - z[45,3] = 0
 R14: - zeta[23,1,1] - zeta[23,1,2] + z[23,1] - z[23,2] - z[23,3] = 0
 R15: - zeta[19,1,1] - zeta[19,1,2] + z[19,1] - z[19,2] - z[19,3] = 0
 R16: - zeta[6,1,1] - zeta[6,1,2] + z[6,1] - z[6,2] - z[6,3] = 0
 R17: - zeta[25,1,1] - zeta[25,1,2] + z[25,1] - z[25,2] - z[25,3] = 0
 R18: - zeta[4,1,1] - zeta[4,1,2] + z[4,1] - z[4,2] - z[4,3] = 0
 R19: - zeta[26,1,1] - zeta[26,1,2] + z[26,1] - z[26,2] - z[26,3] = 0
 R20: - zeta[41,1,1] - zeta[41,1,2] + z[41,1] - z[41,2] - z[41,3] = 0
 R21: - zeta[31,1,1] - zeta[31,1,2] + z[31,1] - z[31,2] - z[31,3] = 0
 R22: - zeta[7,1,1] - zeta[7,1,2] + z[7,1] - z[7,2] - z[7,3] = 0
 R23: - zeta[0,1,1] - zeta[0,1,2] + z[0,1] - z[0,2] - z[0,3] = 0
 R24: - zeta[43,2,1] - zeta[43,2,2] + z[43,2] - z[43,4] - z[43,5] = 0
 R25: - zeta[1,2,1] - zeta[1,2,2] + z[1,2] - z[1,4] - z[1,5] = 0
 R26: - zeta[46,2,1] - zeta[46,2,2] + z[46,2] - z[46,4] - z[46,5] = 0
 R27: - zeta[38,2,1] - zeta[38,2,2] + z[38,2] - z[38,4] - z[38,5] = 0
 R28: - zeta[16,2,1] - zeta[16,2,2] + z[16,2] - z[16,4] - z[16,5] = 0
 R29: - zeta[27,2,1] - zeta[27,2,2] + z[27,2] - z[27,4] - z[27,5] = 0
 R30: - zeta[3,2,1] - zeta[3,2,2] + z[3,2] - z[3,4] - z[3,5] = 0
 R31: - zeta[24,2,1] - zeta[24,2,2] + z[24,2] - z[24,4] - z[24,5] = 0
 R32: - zeta[21,2,1] - zeta[21,2,2] + z[21,2] - z[21,4] - z[21,5] = 0
 R33: - zeta[49,2,1] - zeta[49,2,2] + z[49,2] - z[49,4] - z[49,5] = 0
 R34: - zeta[48,2,1] - zeta[48,2,2] + z[48,2] - z[48,4] - z[48,5] = 0
 R35: - zeta[34,2,1] - zeta[34,2,2] + z[34,2] - z[34,4] - z[34,5] = 0
 R36: - zeta[9,2,1] - zeta[9,2,2] + z[9,2] - z[9,4] - z[9,5] = 0
 R37: - zeta[45,2,1] - zeta[45,2,2] + z[45,2] - z[45,4] - z[45,5] = 0
 R38: - zeta[23,2,1] - zeta[23,2,2] + z[23,2] - z[23,4] - z[23,5] = 0
 R39: - zeta[19,2,1] - zeta[19,2,2] + z[19,2] - z[19,4] - z[19,5] = 0
 R40: - zeta[6,2,1] - zeta[6,2,2] + z[6,2] - z[6,4] - z[6,5] = 0
 R41: - zeta[25,2,1] - zeta[25,2,2] + z[25,2] - z[25,4] - z[25,5] = 0
 R42: - zeta[4,2,1] - zeta[4,2,2] + z[4,2] - z[4,4] - z[4,5] = 0
 R43: - zeta[26,2,1] - zeta[26,2,2] + z[26,2] - z[26,4] - z[26,5] = 0
 R44: - zeta[41,2,1] - zeta[41,2,2] + z[41,2] - z[41,4] - z[41,5] = 0
 R45: - zeta[31,2,1] - zeta[31,2,2] + z[31,2] - z[31,4] - z[31,5] = 0
 R46: - zeta[7,2,1] - zeta[7,2,2] + z[7,2] - z[7,4] - z[7,5] = 0
 R47: - zeta[0,2,1] - zeta[0,2,2] + z[0,2] - z[0,4] - z[0,5] = 0
 R48: - zeta[43,3,1] - zeta[43,3,2] + z[43,3] - z[43,6] - z[43,7] = 0
 R49: - zeta[1,3,1] - zeta[1,3,2] + z[1,3] - z[1,6] - z[1,7] = 0
 R50: - zeta[46,3,1] - zeta[46,3,2] + z[46,3] - z[46,6] - z[46,7] = 0
 R51: - zeta[38,3,1] - zeta[38,3,2] + z[38,3] - z[38,6] - z[38,7] = 0
 R52: - zeta[16,3,1] - zeta[16,3,2] + z[16,3] - z[16,6] - z[16,7] = 0
 R53: - zeta[27,3,1] - zeta[27,3,2] + z[27,3] - z[27,6] - z[27,7] = 0
 R54: - zeta[3,3,1] - zeta[3,3,2] + z[3,3] - z[3,6] - z[3,7] = 0
 R55: - zeta[24,3,1] - zeta[24,3,2] + z[24,3] - z[24,6] - z[24,7] = 0
 R56: - zeta[21,3,1] - zeta[21,3,2] + z[21,3] - z[21,6] - z[21,7] = 0
 R57: - zeta[49,3,1] - zeta[49,3,2] + z[49,3] - z[49,6] - z[49,7] = 0
 R58: - zeta[48,3,1] - zeta[48,3,2] + z[48,3] - z[48,6] - z[48,7] = 0
 R59: - zeta[34,3,1] - zeta[34,3,2] + z[34,3] - z[34,6] - z[34,7] = 0
 R60: - zeta[9,3,1] - zeta[9,3,2] + z[9,3] - z[9,6] - z[9,7] = 0
 R61: - zeta[45,3,1] - zeta[45,3,2] + z[45,3] - z[45,6] - z[45,7] = 0
 R62: - zeta[23,3,1] - zeta[23,3,2] + z[23,3] - z[23,6] - z[23,7] = 0
 R63: - zeta[19,3,1] - zeta[19,3,2] + z[19,3] - z[19,6] - z[19,7] = 0
 R64: - zeta[6,3,1] - zeta[6,3,2] + z[6,3] - z[6,6] - z[6,7] = 0
 R65: - zeta[25,3,1] - zeta[25,3,2] + z[25,3] - z[25,6] - z[25,7] = 0
 R66: - zeta[4,3,1] - zeta[4,3,2] + z[4,3] - z[4,6] - z[4,7] = 0
 R67: - zeta[26,3,1] - zeta[26,3,2] + z[26,3] - z[26,6] - z[26,7] = 0
 R68: - zeta[41,3,1] - zeta[41,3,2] + z[41,3] - z[41,6] - z[41,7] = 0
 R69: - zeta[31,3,1] - zeta[31,3,2] + z[31,3] - z[31,6] - z[31,7] = 0
 R70: - zeta[7,3,1] - zeta[7,3,2] + z[7,3] - z[7,6] - z[7,7] = 0
 R71: - zeta[0,3,1] - zeta[0,3,2] + z[0,3] - z[0,6] - z[0,7] = 0
 R72: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] + z[43,2] <= 0
 R73: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.3] - b[2,race_factor.4] - b[2,race_factor.5]
   - b[2,race_factor.6] + z[43,4] <= 0
 R74: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.3] - b[3,race_factor.4] - b[3,race_factor.5]
   - b[3,race_factor.6] + z[43,6] <= 0
 R75: - b[1,two_year_recid.0] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] + z[1,2] <= 0
 R76: - b[2,two_year_recid.0] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.1] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] + z[1,4] <= 0
 R77: - b[3,two_year_recid.0] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.1] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] + z[1,6] <= 0
 R78: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] + z[46,2] <= 0
 R79: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.3] - b[2,race_factor.4] - b[2,race_factor.5]
   - b[2,race_factor.6] + z[46,4] <= 0
 R80: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.3] - b[3,race_factor.4] - b[3,race_factor.5]
   - b[3,race_factor.6] + z[46,6] <= 0
 R81: - b[1,priors_count.1] - b[1,gender_factor.1] - b[1,age_factor.1]
   - b[1,age_factor.3] - b[1,race_factor.1] - b[1,race_factor.3]
   - b[1,race_factor.4] - b[1,race_factor.5] - b[1,race_factor.6] + z[38,2]
   <= 0
 R82: - b[2,priors_count.1] - b[2,gender_factor.1] - b[2,age_factor.1]
   - b[2,age_factor.3] - b[2,race_factor.1] - b[2,race_factor.3]
   - b[2,race_factor.4] - b[2,race_factor.5] - b[2,race_factor.6] + z[38,4]
   <= 0
 R83: - b[3,priors_count.1] - b[3,gender_factor.1] - b[3,age_factor.1]
   - b[3,age_factor.3] - b[3,race_factor.1] - b[3,race_factor.3]
   - b[3,race_factor.4] - b[3,race_factor.5] - b[3,race_factor.6] + z[38,6]
   <= 0
 R84: - b[1,priors_count.1] - b[1,two_year_recid.0] - b[1,age_factor.2]
   - b[1,age_factor.3] - b[1,race_factor.2] - b[1,race_factor.3]
   - b[1,race_factor.4] - b[1,race_factor.5] - b[1,race_factor.6] + z[16,2]
   <= 0
 R85: - b[2,priors_count.1] - b[2,two_year_recid.0] - b[2,age_factor.2]
   - b[2,age_factor.3] - b[2,race_factor.2] - b[2,race_factor.3]
   - b[2,race_factor.4] - b[2,race_factor.5] - b[2,race_factor.6] + z[16,4]
   <= 0
 R86: - b[3,priors_count.1] - b[3,two_year_recid.0] - b[3,age_factor.2]
   - b[3,age_factor.3] - b[3,race_factor.2] - b[3,race_factor.3]
   - b[3,race_factor.4] - b[3,race_factor.5] - b[3,race_factor.6] + z[16,6]
   <= 0
 R87: - b[1,priors_count.1] - b[1,two_year_recid.0] - b[1,age_factor.2]
   - b[1,age_factor.3] - b[1,race_factor.2] - b[1,race_factor.3]
   - b[1,race_factor.4] - b[1,race_factor.5] - b[1,race_factor.6] + z[27,2]
   <= 0
 R88: - b[2,priors_count.1] - b[2,two_year_recid.0] - b[2,age_factor.2]
   - b[2,age_factor.3] - b[2,race_factor.2] - b[2,race_factor.3]
   - b[2,race_factor.4] - b[2,race_factor.5] - b[2,race_factor.6] + z[27,4]
   <= 0
 R89: - b[3,priors_count.1] - b[3,two_year_recid.0] - b[3,age_factor.2]
   - b[3,age_factor.3] - b[3,race_factor.2] - b[3,race_factor.3]
   - b[3,race_factor.4] - b[3,race_factor.5] - b[3,race_factor.6] + z[27,6]
   <= 0
 R90: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,crime_factor.1] + z[3,2] <= 0
 R91: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,crime_factor.1] + z[3,4] <= 0
 R92: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,crime_factor.1] + z[3,6] <= 0
 R93: - b[1,priors_count.1] - b[1,two_year_recid.0] - b[1,age_factor.2]
   - b[1,age_factor.3] - b[1,race_factor.1] - b[1,race_factor.3]
   - b[1,race_factor.4] - b[1,race_factor.5] - b[1,race_factor.6] + z[24,2]
   <= 0
 R94: - b[2,priors_count.1] - b[2,two_year_recid.0] - b[2,age_factor.2]
   - b[2,age_factor.3] - b[2,race_factor.1] - b[2,race_factor.3]
   - b[2,race_factor.4] - b[2,race_factor.5] - b[2,race_factor.6] + z[24,4]
   <= 0
 R95: - b[3,priors_count.1] - b[3,two_year_recid.0] - b[3,age_factor.2]
   - b[3,age_factor.3] - b[3,race_factor.1] - b[3,race_factor.3]
   - b[3,race_factor.4] - b[3,race_factor.5] - b[3,race_factor.6] + z[24,6]
   <= 0
 R96: - b[1,two_year_recid.0] - b[1,age_factor.1] - b[1,age_factor.2]
   - b[1,race_factor.1] - b[1,race_factor.2] - b[1,race_factor.3]
   - b[1,race_factor.5] - b[1,race_factor.6] + z[21,2] <= 0
 R97: - b[2,two_year_recid.0] - b[2,age_factor.1] - b[2,age_factor.2]
   - b[2,race_factor.1] - b[2,race_factor.2] - b[2,race_factor.3]
   - b[2,race_factor.5] - b[2,race_factor.6] + z[21,4] <= 0
 R98: - b[3,two_year_recid.0] - b[3,age_factor.1] - b[3,age_factor.2]
   - b[3,race_factor.1] - b[3,race_factor.2] - b[3,race_factor.3]
   - b[3,race_factor.5] - b[3,race_factor.6] + z[21,6] <= 0
 R99: - b[1,age_factor.1] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] - b[1,crime_factor.1] + z[49,2] <= 0
 R100: - b[2,age_factor.1] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.3] - b[2,race_factor.4] - b[2,race_factor.5]
   - b[2,race_factor.6] - b[2,crime_factor.1] + z[49,4] <= 0
 R101: - b[3,age_factor.1] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.3] - b[3,race_factor.4] - b[3,race_factor.5]
   - b[3,race_factor.6] - b[3,crime_factor.1] + z[49,6] <= 0
 R102: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.5]
   - b[1,race_factor.6] - b[1,crime_factor.1] + z[48,2] <= 0
 R103: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.5]
   - b[2,race_factor.6] - b[2,crime_factor.1] + z[48,4] <= 0
 R104: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.5]
   - b[3,race_factor.6] - b[3,crime_factor.1] + z[48,6] <= 0
 R105: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.2]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] - b[1,crime_factor.1] + z[34,2] <= 0
 R106: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.2]
   - b[2,race_factor.3] - b[2,race_factor.4] - b[2,race_factor.5]
   - b[2,race_factor.6] - b[2,crime_factor.1] + z[34,4] <= 0
 R107: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.2]
   - b[3,race_factor.3] - b[3,race_factor.4] - b[3,race_factor.5]
   - b[3,race_factor.6] - b[3,crime_factor.1] + z[34,6] <= 0
 R108: - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[9,2]
   <= 0
 R109: - b[2,gender_factor.1] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1] + z[9,4]
   <= 0
 R110: - b[3,gender_factor.1] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1] + z[9,6]
   <= 0
 R111: - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.1] - b[1,race_factor.2] - b[1,race_factor.3]
   - b[1,race_factor.4] - b[1,race_factor.5] + z[45,2] <= 0
 R112: - b[2,gender_factor.1] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.1] - b[2,race_factor.2] - b[2,race_factor.3]
   - b[2,race_factor.4] - b[2,race_factor.5] + z[45,4] <= 0
 R113: - b[3,gender_factor.1] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.1] - b[3,race_factor.2] - b[3,race_factor.3]
   - b[3,race_factor.4] - b[3,race_factor.5] + z[45,6] <= 0
 R114: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,crime_factor.1] + z[23,2] <= 0
 R115: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,crime_factor.1] + z[23,4] <= 0
 R116: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,crime_factor.1] + z[23,6] <= 0
 R117: - b[1,two_year_recid.0] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1]
   + z[19,2] <= 0
 R118: - b[2,two_year_recid.0] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1]
   + z[19,4] <= 0
 R119: - b[3,two_year_recid.0] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1]
   + z[19,6] <= 0
 R120: - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] - b[1,crime_factor.1] + z[6,2]
   <= 0
 R121: - b[2,gender_factor.1] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] - b[2,crime_factor.1] + z[6,4]
   <= 0
 R122: - b[3,gender_factor.1] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] - b[3,crime_factor.1] + z[6,6]
   <= 0
 R123: - b[1,priors_count.1] - b[1,two_year_recid.0] - b[1,age_factor.2]
   - b[1,age_factor.3] - b[1,race_factor.1] - b[1,race_factor.3]
   - b[1,race_factor.4] - b[1,race_factor.5] - b[1,race_factor.6] + z[25,2]
   <= 0
 R124: - b[2,priors_count.1] - b[2,two_year_recid.0] - b[2,age_factor.2]
   - b[2,age_factor.3] - b[2,race_factor.1] - b[2,race_factor.3]
   - b[2,race_factor.4] - b[2,race_factor.5] - b[2,race_factor.6] + z[25,4]
   <= 0
 R125: - b[3,priors_count.1] - b[3,two_year_recid.0] - b[3,age_factor.2]
   - b[3,age_factor.3] - b[3,race_factor.1] - b[3,race_factor.3]
   - b[3,race_factor.4] - b[3,race_factor.5] - b[3,race_factor.6] + z[25,6]
   <= 0
 R126: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,two_year_recid.0]
   - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.2]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] + z[4,2] <= 0
 R127: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,two_year_recid.0]
   - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.2]
   - b[2,race_factor.3] - b[2,race_factor.4] - b[2,race_factor.5]
   - b[2,race_factor.6] + z[4,4] <= 0
 R128: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,two_year_recid.0]
   - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.2]
   - b[3,race_factor.3] - b[3,race_factor.4] - b[3,race_factor.5]
   - b[3,race_factor.6] + z[4,6] <= 0
 R129: - b[1,two_year_recid.0] - b[1,age_factor.2] - b[1,age_factor.3]
   - b[1,race_factor.1] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] - b[1,race_factor.6] + z[26,2] <= 0
 R130: - b[2,two_year_recid.0] - b[2,age_factor.2] - b[2,age_factor.3]
   - b[2,race_factor.1] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] - b[2,race_factor.6] + z[26,4] <= 0
 R131: - b[3,two_year_recid.0] - b[3,age_factor.2] - b[3,age_factor.3]
   - b[3,race_factor.1] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] - b[3,race_factor.6] + z[26,6] <= 0
 R132: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,two_year_recid.0]
   - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] + z[41,2] <= 0
 R133: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,two_year_recid.0]
   - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.3] - b[2,race_factor.4] - b[2,race_factor.5]
   - b[2,race_factor.6] + z[41,4] <= 0
 R134: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,two_year_recid.0]
   - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.3] - b[3,race_factor.4] - b[3,race_factor.5]
   - b[3,race_factor.6] + z[41,6] <= 0
 R135: - b[1,two_year_recid.0] - b[1,gender_factor.1] - b[1,age_factor.2]
   - b[1,age_factor.3] - b[1,race_factor.2] - b[1,race_factor.3]
   - b[1,race_factor.4] - b[1,race_factor.5] - b[1,race_factor.6]
   - b[1,crime_factor.1] + z[31,2] <= 0
 R136: - b[2,two_year_recid.0] - b[2,gender_factor.1] - b[2,age_factor.2]
   - b[2,age_factor.3] - b[2,race_factor.2] - b[2,race_factor.3]
   - b[2,race_factor.4] - b[2,race_factor.5] - b[2,race_factor.6]
   - b[2,crime_factor.1] + z[31,4] <= 0
 R137: - b[3,two_year_recid.0] - b[3,gender_factor.1] - b[3,age_factor.2]
   - b[3,age_factor.3] - b[3,race_factor.2] - b[3,race_factor.3]
   - b[3,race_factor.4] - b[3,race_factor.5] - b[3,race_factor.6]
   - b[3,crime_factor.1] + z[31,6] <= 0
 R138: - b[1,age_factor.2] - b[1,age_factor.3] - b[1,race_factor.2]
   - b[1,race_factor.3] - b[1,race_factor.4] - b[1,race_factor.5]
   - b[1,race_factor.6] + z[7,2] <= 0
 R139: - b[2,age_factor.2] - b[2,age_factor.3] - b[2,race_factor.2]
   - b[2,race_factor.3] - b[2,race_factor.4] - b[2,race_factor.5]
   - b[2,race_factor.6] + z[7,4] <= 0
 R140: - b[3,age_factor.2] - b[3,age_factor.3] - b[3,race_factor.2]
   - b[3,race_factor.3] - b[3,race_factor.4] - b[3,race_factor.5]
   - b[3,race_factor.6] + z[7,6] <= 0
 R141: - b[1,age_factor.1] - b[1,age_factor.3] - b[1,race_factor.1]
   - b[1,race_factor.2] - b[1,race_factor.3] - b[1,race_factor.4]
   - b[1,race_factor.5] + z[0,2] <= 0
 R142: - b[2,age_factor.1] - b[2,age_factor.3] - b[2,race_factor.1]
   - b[2,race_factor.2] - b[2,race_factor.3] - b[2,race_factor.4]
   - b[2,race_factor.5] + z[0,4] <= 0
 R143: - b[3,age_factor.1] - b[3,age_factor.3] - b[3,race_factor.1]
   - b[3,race_factor.2] - b[3,race_factor.3] - b[3,race_factor.4]
   - b[3,race_factor.5] + z[0,6] <= 0
 R144: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.2]
   - b[1,crime_factor.1] + z[43,3] <= 0
 R145: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.2]
   - b[2,crime_factor.1] + z[43,5] <= 0
 R146: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.2]
   - b[3,crime_factor.1] + z[43,7] <= 0
 R147: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.1] - b[1,race_factor.2] - b[1,crime_factor.1] + z[1,3]
   <= 0
 R148: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.1] - b[2,race_factor.2] - b[2,crime_factor.1] + z[1,5]
   <= 0
 R149: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.1] - b[3,race_factor.2] - b[3,crime_factor.1] + z[1,7]
   <= 0
 R150: - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.2]
   - b[1,crime_factor.1] + z[46,3] <= 0
 R151: - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.2]
   - b[2,crime_factor.1] + z[46,5] <= 0
 R152: - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.2]
   - b[3,crime_factor.1] + z[46,7] <= 0
 R153: - b[1,priors_count.2] - b[1,priors_count.3] - b[1,priors_count.4]
   - b[1,priors_count.5] - b[1,priors_count.6] - b[1,priors_count.7]
   - b[1,priors_count.8] - b[1,two_year_recid.0] - b[1,age_factor.2]
   - b[1,race_factor.2] - b[1,crime_factor.1] + z[38,3] <= 0
 R154: - b[2,priors_count.2] - b[2,priors_count.3] - b[2,priors_count.4]
   - b[2,priors_count.5] - b[2,priors_count.6] - b[2,priors_count.7]
   - b[2,priors_count.8] - b[2,two_year_recid.0] - b[2,age_factor.2]
   - b[2,race_factor.2] - b[2,crime_factor.1] + z[38,5] <= 0
 R155: - b[3,priors_count.2] - b[3,priors_count.3] - b[3,priors_count.4]
   - b[3,priors_count.5] - b[3,priors_count.6] - b[3,priors_count.7]
   - b[3,priors_count.8] - b[3,two_year_recid.0] - b[3,age_factor.2]
   - b[3,race_factor.2] - b[3,crime_factor.1] + z[38,7] <= 0
 R156: - b[1,priors_count.2] - b[1,priors_count.3] - b[1,priors_count.4]
   - b[1,priors_count.5] - b[1,priors_count.6] - b[1,priors_count.7]
   - b[1,priors_count.8] - b[1,gender_factor.1] - b[1,age_factor.1]
   - b[1,race_factor.1] - b[1,crime_factor.1] + z[16,3] <= 0
 R157: - b[2,priors_count.2] - b[2,priors_count.3] - b[2,priors_count.4]
   - b[2,priors_count.5] - b[2,priors_count.6] - b[2,priors_count.7]
   - b[2,priors_count.8] - b[2,gender_factor.1] - b[2,age_factor.1]
   - b[2,race_factor.1] - b[2,crime_factor.1] + z[16,5] <= 0
 R158: - b[3,priors_count.2] - b[3,priors_count.3] - b[3,priors_count.4]
   - b[3,priors_count.5] - b[3,priors_count.6] - b[3,priors_count.7]
   - b[3,priors_count.8] - b[3,gender_factor.1] - b[3,age_factor.1]
   - b[3,race_factor.1] - b[3,crime_factor.1] + z[16,7] <= 0
 R159: - b[1,priors_count.2] - b[1,priors_count.3] - b[1,priors_count.4]
   - b[1,priors_count.5] - b[1,priors_count.6] - b[1,priors_count.7]
   - b[1,priors_count.8] - b[1,gender_factor.1] - b[1,age_factor.1]
   - b[1,race_factor.1] - b[1,crime_factor.1] + z[27,3] <= 0
 R160: - b[2,priors_count.2] - b[2,priors_count.3] - b[2,priors_count.4]
   - b[2,priors_count.5] - b[2,priors_count.6] - b[2,priors_count.7]
   - b[2,priors_count.8] - b[2,gender_factor.1] - b[2,age_factor.1]
   - b[2,race_factor.1] - b[2,crime_factor.1] + z[27,5] <= 0
 R161: - b[3,priors_count.2] - b[3,priors_count.3] - b[3,priors_count.4]
   - b[3,priors_count.5] - b[3,priors_count.6] - b[3,priors_count.7]
   - b[3,priors_count.8] - b[3,gender_factor.1] - b[3,age_factor.1]
   - b[3,race_factor.1] - b[3,crime_factor.1] + z[27,7] <= 0
 R162: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.6] + z[3,3]
   <= 0
 R163: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.6] + z[3,5]
   <= 0
 R164: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.6] + z[3,7]
   <= 0
 R165: - b[1,priors_count.2] - b[1,priors_count.3] - b[1,priors_count.4]
   - b[1,priors_count.5] - b[1,priors_count.6] - b[1,priors_count.7]
   - b[1,priors_count.8] - b[1,gender_factor.1] - b[1,age_factor.1]
   - b[1,race_factor.2] - b[1,crime_factor.1] + z[24,3] <= 0
 R166: - b[2,priors_count.2] - b[2,priors_count.3] - b[2,priors_count.4]
   - b[2,priors_count.5] - b[2,priors_count.6] - b[2,priors_count.7]
   - b[2,priors_count.8] - b[2,gender_factor.1] - b[2,age_factor.1]
   - b[2,race_factor.2] - b[2,crime_factor.1] + z[24,5] <= 0
 R167: - b[3,priors_count.2] - b[3,priors_count.3] - b[3,priors_count.4]
   - b[3,priors_count.5] - b[3,priors_count.6] - b[3,priors_count.7]
   - b[3,priors_count.8] - b[3,gender_factor.1] - b[3,age_factor.1]
   - b[3,race_factor.2] - b[3,crime_factor.1] + z[24,7] <= 0
 R168: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.3] - b[1,race_factor.4] - b[1,crime_factor.1] + z[21,3]
   <= 0
 R169: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.3] - b[2,race_factor.4] - b[2,crime_factor.1] + z[21,5]
   <= 0
 R170: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.3] - b[3,race_factor.4] - b[3,crime_factor.1] + z[21,7]
   <= 0
 R171: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,race_factor.2]
   + z[49,3] <= 0
 R172: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.2] - b[2,race_factor.2]
   + z[49,5] <= 0
 R173: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.2] - b[3,race_factor.2]
   + z[49,7] <= 0
 R174: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.4]
   + z[48,3] <= 0
 R175: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.4]
   + z[48,5] <= 0
 R176: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.4]
   + z[48,7] <= 0
 R177: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.1]
   + z[34,3] <= 0
 R178: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.1]
   + z[34,5] <= 0
 R179: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.1]
   + z[34,7] <= 0
 R180: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,age_factor.1] - b[1,race_factor.1] + z[9,3] <= 0
 R181: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,age_factor.1] - b[2,race_factor.1] + z[9,5] <= 0
 R182: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,age_factor.1] - b[3,race_factor.1] + z[9,7] <= 0
 R183: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,age_factor.1] - b[1,race_factor.6] - b[1,crime_factor.1] + z[45,3]
   <= 0
 R184: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,age_factor.1] - b[2,race_factor.6] - b[2,crime_factor.1] + z[45,5]
   <= 0
 R185: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,age_factor.1] - b[3,race_factor.6] - b[3,crime_factor.1] + z[45,7]
   <= 0
 R186: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.6]
   + z[23,3] <= 0
 R187: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.6]
   + z[23,5] <= 0
 R188: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.6]
   + z[23,7] <= 0
 R189: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.1] - b[1,race_factor.1] + z[19,3] <= 0
 R190: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.1] - b[2,race_factor.1] + z[19,5] <= 0
 R191: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.1] - b[3,race_factor.1] + z[19,7] <= 0
 R192: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,age_factor.1] - b[1,race_factor.1] + z[6,3] <= 0
 R193: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,age_factor.1] - b[2,race_factor.1] + z[6,5] <= 0
 R194: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,age_factor.1] - b[3,race_factor.1] + z[6,7] <= 0
 R195: - b[1,priors_count.2] - b[1,priors_count.3] - b[1,priors_count.4]
   - b[1,priors_count.5] - b[1,priors_count.6] - b[1,priors_count.7]
   - b[1,priors_count.8] - b[1,gender_factor.1] - b[1,age_factor.1]
   - b[1,race_factor.2] - b[1,crime_factor.1] + z[25,3] <= 0
 R196: - b[2,priors_count.2] - b[2,priors_count.3] - b[2,priors_count.4]
   - b[2,priors_count.5] - b[2,priors_count.6] - b[2,priors_count.7]
   - b[2,priors_count.8] - b[2,gender_factor.1] - b[2,age_factor.1]
   - b[2,race_factor.2] - b[2,crime_factor.1] + z[25,5] <= 0
 R197: - b[3,priors_count.2] - b[3,priors_count.3] - b[3,priors_count.4]
   - b[3,priors_count.5] - b[3,priors_count.6] - b[3,priors_count.7]
   - b[3,priors_count.8] - b[3,gender_factor.1] - b[3,age_factor.1]
   - b[3,race_factor.2] - b[3,crime_factor.1] + z[25,7] <= 0
 R198: - b[1,priors_count.3] - b[1,priors_count.4] - b[1,priors_count.5]
   - b[1,priors_count.6] - b[1,priors_count.7] - b[1,priors_count.8]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.1]
   - b[1,crime_factor.1] + z[4,3] <= 0
 R199: - b[2,priors_count.3] - b[2,priors_count.4] - b[2,priors_count.5]
   - b[2,priors_count.6] - b[2,priors_count.7] - b[2,priors_count.8]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.1]
   - b[2,crime_factor.1] + z[4,5] <= 0
 R200: - b[3,priors_count.3] - b[3,priors_count.4] - b[3,priors_count.5]
   - b[3,priors_count.6] - b[3,priors_count.7] - b[3,priors_count.8]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.1]
   - b[3,crime_factor.1] + z[4,7] <= 0
 R201: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,gender_factor.1]
   - b[1,age_factor.1] - b[1,race_factor.2] - b[1,crime_factor.1] + z[26,3]
   <= 0
 R202: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,gender_factor.1]
   - b[2,age_factor.1] - b[2,race_factor.2] - b[2,crime_factor.1] + z[26,5]
   <= 0
 R203: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,gender_factor.1]
   - b[3,age_factor.1] - b[3,race_factor.2] - b[3,crime_factor.1] + z[26,7]
   <= 0
 R204: - b[1,priors_count.3] - b[1,priors_count.4] - b[1,priors_count.5]
   - b[1,priors_count.6] - b[1,priors_count.7] - b[1,priors_count.8]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.2]
   - b[1,crime_factor.1] + z[41,3] <= 0
 R205: - b[2,priors_count.3] - b[2,priors_count.4] - b[2,priors_count.5]
   - b[2,priors_count.6] - b[2,priors_count.7] - b[2,priors_count.8]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.2]
   - b[2,crime_factor.1] + z[41,5] <= 0
 R206: - b[3,priors_count.3] - b[3,priors_count.4] - b[3,priors_count.5]
   - b[3,priors_count.6] - b[3,priors_count.7] - b[3,priors_count.8]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.2]
   - b[3,crime_factor.1] + z[41,7] <= 0
 R207: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,age_factor.1]
   - b[1,race_factor.1] + z[31,3] <= 0
 R208: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,age_factor.1]
   - b[2,race_factor.1] + z[31,5] <= 0
 R209: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,age_factor.1]
   - b[3,race_factor.1] + z[31,7] <= 0
 R210: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.1] - b[1,race_factor.1]
   - b[1,crime_factor.1] + z[7,3] <= 0
 R211: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.1] - b[2,race_factor.1]
   - b[2,crime_factor.1] + z[7,5] <= 0
 R212: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.1] - b[3,race_factor.1]
   - b[3,crime_factor.1] + z[7,7] <= 0
 R213: - b[1,priors_count.1] - b[1,priors_count.2] - b[1,priors_count.3]
   - b[1,priors_count.4] - b[1,priors_count.5] - b[1,priors_count.6]
   - b[1,priors_count.7] - b[1,priors_count.8] - b[1,two_year_recid.0]
   - b[1,gender_factor.1] - b[1,age_factor.2] - b[1,race_factor.6]
   - b[1,crime_factor.1] + z[0,3] <= 0
 R214: - b[2,priors_count.1] - b[2,priors_count.2] - b[2,priors_count.3]
   - b[2,priors_count.4] - b[2,priors_count.5] - b[2,priors_count.6]
   - b[2,priors_count.7] - b[2,priors_count.8] - b[2,two_year_recid.0]
   - b[2,gender_factor.1] - b[2,age_factor.2] - b[2,race_factor.6]
   - b[2,crime_factor.1] + z[0,5] <= 0
 R215: - b[3,priors_count.1] - b[3,priors_count.2] - b[3,priors_count.3]
   - b[3,priors_count.4] - b[3,priors_count.5] - b[3,priors_count.6]
   - b[3,priors_count.7] - b[3,priors_count.8] - b[3,two_year_recid.0]
   - b[3,gender_factor.1] - b[3,age_factor.2] - b[3,race_factor.6]
   - b[3,crime_factor.1] + z[0,7] <= 0
 R216: b[1,priors_count.1] + b[1,priors_count.2] + b[1,priors_count.3]
   + b[1,priors_count.4] + b[1,priors_count.5] + b[1,priors_count.6]
   + b[1,priors_count.7] + b[1,priors_count.8] + b[1,two_year_recid.0]
   + b[1,gender_factor.1] + b[1,age_factor.1] + b[1,age_factor.2]
   + b[1,age_factor.3] + b[1,race_factor.1] + b[1,race_factor.2]
   + b[1,race_factor.3] + b[1,race_factor.4] + b[1,race_factor.5]
   + b[1,race_factor.6] + b[1,crime_factor.1] + p[1] = 1
 R217: b[2,priors_count.1] + b[2,priors_count.2] + b[2,priors_count.3]
   + b[2,priors_count.4] + b[2,priors_count.5] + b[2,priors_count.6]
   + b[2,priors_count.7] + b[2,priors_count.8] + b[2,two_year_recid.0]
   + b[2,gender_factor.1] + b[2,age_factor.1] + b[2,age_factor.2]
   + b[2,age_factor.3] + b[2,race_factor.1] + b[2,race_factor.2]
   + b[2,race_factor.3] + b[2,race_factor.4] + b[2,race_factor.5]
   + b[2,race_factor.6] + b[2,crime_factor.1] + p[1] + p[2] = 1
 R218: b[3,priors_count.1] + b[3,priors_count.2] + b[3,priors_count.3]
   + b[3,priors_count.4] + b[3,priors_count.5] + b[3,priors_count.6]
   + b[3,priors_count.7] + b[3,priors_count.8] + b[3,two_year_recid.0]
   + b[3,gender_factor.1] + b[3,age_factor.1] + b[3,age_factor.2]
   + b[3,age_factor.3] + b[3,race_factor.1] + b[3,race_factor.2]
   + b[3,race_factor.3] + b[3,race_factor.4] + b[3,race_factor.5]
   + b[3,race_factor.6] + b[3,crime_factor.1] + p[1] + p[3] = 1
 R219: p[1] + p[2] + p[4] = 1
 R220: p[1] + p[2] + p[5] = 1
 R221: p[1] + p[3] + p[6] = 1
 R222: p[1] + p[3] + p[7] = 1
 R223: - beta[1,1] + zeta[43,1,1] <= 0
 R224: - beta[1,2] + zeta[43,1,2] <= 0
 R225: - beta[1,1] + zeta[1,1,1] <= 0
 R226: - beta[1,2] + zeta[1,1,2] <= 0
 R227: - beta[1,1] + zeta[46,1,1] <= 0
 R228: - beta[1,2] + zeta[46,1,2] <= 0
 R229: - beta[1,1] + zeta[38,1,1] <= 0
 R230: - beta[1,2] + zeta[38,1,2] <= 0
 R231: - beta[1,1] + zeta[16,1,1] <= 0
 R232: - beta[1,2] + zeta[16,1,2] <= 0
 R233: - beta[1,1] + zeta[27,1,1] <= 0
 R234: - beta[1,2] + zeta[27,1,2] <= 0
 R235: - beta[1,1] + zeta[3,1,1] <= 0
 R236: - beta[1,2] + zeta[3,1,2] <= 0
 R237: - beta[1,1] + zeta[24,1,1] <= 0
 R238: - beta[1,2] + zeta[24,1,2] <= 0
 R239: - beta[1,1] + zeta[21,1,1] <= 0
 R240: - beta[1,2] + zeta[21,1,2] <= 0
 R241: - beta[1,1] + zeta[49,1,1] <= 0
 R242: - beta[1,2] + zeta[49,1,2] <= 0
 R243: - beta[1,1] + zeta[48,1,1] <= 0
 R244: - beta[1,2] + zeta[48,1,2] <= 0
 R245: - beta[1,1] + zeta[34,1,1] <= 0
 R246: - beta[1,2] + zeta[34,1,2] <= 0
 R247: - beta[1,1] + zeta[9,1,1] <= 0
 R248: - beta[1,2] + zeta[9,1,2] <= 0
 R249: - beta[1,1] + zeta[45,1,1] <= 0
 R250: - beta[1,2] + zeta[45,1,2] <= 0
 R251: - beta[1,1] + zeta[23,1,1] <= 0
 R252: - beta[1,2] + zeta[23,1,2] <= 0
 R253: - beta[1,1] + zeta[19,1,1] <= 0
 R254: - beta[1,2] + zeta[19,1,2] <= 0
 R255: - beta[1,1] + zeta[6,1,1] <= 0
 R256: - beta[1,2] + zeta[6,1,2] <= 0
 R257: - beta[1,1] + zeta[25,1,1] <= 0
 R258: - beta[1,2] + zeta[25,1,2] <= 0
 R259: - beta[1,1] + zeta[4,1,1] <= 0
 R260: - beta[1,2] + zeta[4,1,2] <= 0
 R261: - beta[1,1] + zeta[26,1,1] <= 0
 R262: - beta[1,2] + zeta[26,1,2] <= 0
 R263: - beta[1,1] + zeta[41,1,1] <= 0
 R264: - beta[1,2] + zeta[41,1,2] <= 0
 R265: - beta[1,1] + zeta[31,1,1] <= 0
 R266: - beta[1,2] + zeta[31,1,2] <= 0
 R267: - beta[1,1] + zeta[7,1,1] <= 0
 R268: - beta[1,2] + zeta[7,1,2] <= 0
 R269: - beta[1,1] + zeta[0,1,1] <= 0
 R270: - beta[1,2] + zeta[0,1,2] <= 0
 R271: - beta[2,1] + zeta[43,2,1] <= 0
 R272: - beta[2,2] + zeta[43,2,2] <= 0
 R273: - beta[2,1] + zeta[1,2,1] <= 0
 R274: - beta[2,2] + zeta[1,2,2] <= 0
 R275: - beta[2,1] + zeta[46,2,1] <= 0
 R276: - beta[2,2] + zeta[46,2,2] <= 0
 R277: - beta[2,1] + zeta[38,2,1] <= 0
 R278: - beta[2,2] + zeta[38,2,2] <= 0
 R279: - beta[2,1] + zeta[16,2,1] <= 0
 R280: - beta[2,2] + zeta[16,2,2] <= 0
 R281: - beta[2,1] + zeta[27,2,1] <= 0
 R282: - beta[2,2] + zeta[27,2,2] <= 0
 R283: - beta[2,1] + zeta[3,2,1] <= 0
 R284: - beta[2,2] + zeta[3,2,2] <= 0
 R285: - beta[2,1] + zeta[24,2,1] <= 0
 R286: - beta[2,2] + zeta[24,2,2] <= 0
 R287: - beta[2,1] + zeta[21,2,1] <= 0
 R288: - beta[2,2] + zeta[21,2,2] <= 0
 R289: - beta[2,1] + zeta[49,2,1] <= 0
 R290: - beta[2,2] + zeta[49,2,2] <= 0
 R291: - beta[2,1] + zeta[48,2,1] <= 0
 R292: - beta[2,2] + zeta[48,2,2] <= 0
 R293: - beta[2,1] + zeta[34,2,1] <= 0
 R294: - beta[2,2] + zeta[34,2,2] <= 0
 R295: - beta[2,1] + zeta[9,2,1] <= 0
 R296: - beta[2,2] + zeta[9,2,2] <= 0
 R297: - beta[2,1] + zeta[45,2,1] <= 0
 R298: - beta[2,2] + zeta[45,2,2] <= 0
 R299: - beta[2,1] + zeta[23,2,1] <= 0
 R300: - beta[2,2] + zeta[23,2,2] <= 0
 R301: - beta[2,1] + zeta[19,2,1] <= 0
 R302: - beta[2,2] + zeta[19,2,2] <= 0
 R303: - beta[2,1] + zeta[6,2,1] <= 0
 R304: - beta[2,2] + zeta[6,2,2] <= 0
 R305: - beta[2,1] + zeta[25,2,1] <= 0
 R306: - beta[2,2] + zeta[25,2,2] <= 0
 R307: - beta[2,1] + zeta[4,2,1] <= 0
 R308: - beta[2,2] + zeta[4,2,2] <= 0
 R309: - beta[2,1] + zeta[26,2,1] <= 0
 R310: - beta[2,2] + zeta[26,2,2] <= 0
 R311: - beta[2,1] + zeta[41,2,1] <= 0
 R312: - beta[2,2] + zeta[41,2,2] <= 0
 R313: - beta[2,1] + zeta[31,2,1] <= 0
 R314: - beta[2,2] + zeta[31,2,2] <= 0
 R315: - beta[2,1] + zeta[7,2,1] <= 0
 R316: - beta[2,2] + zeta[7,2,2] <= 0
 R317: - beta[2,1] + zeta[0,2,1] <= 0
 R318: - beta[2,2] + zeta[0,2,2] <= 0
 R319: - beta[3,1] + zeta[43,3,1] <= 0
 R320: - beta[3,2] + zeta[43,3,2] <= 0
 R321: - beta[3,1] + zeta[1,3,1] <= 0
 R322: - beta[3,2] + zeta[1,3,2] <= 0
 R323: - beta[3,1] + zeta[46,3,1] <= 0
 R324: - beta[3,2] + zeta[46,3,2] <= 0
 R325: - beta[3,1] + zeta[38,3,1] <= 0
 R326: - beta[3,2] + zeta[38,3,2] <= 0
 R327: - beta[3,1] + zeta[16,3,1] <= 0
 R328: - beta[3,2] + zeta[16,3,2] <= 0
 R329: - beta[3,1] + zeta[27,3,1] <= 0
 R330: - beta[3,2] + zeta[27,3,2] <= 0
 R331: - beta[3,1] + zeta[3,3,1] <= 0
 R332: - beta[3,2] + zeta[3,3,2] <= 0
 R333: - beta[3,1] + zeta[24,3,1] <= 0
 R334: - beta[3,2] + zeta[24,3,2] <= 0
 R335: - beta[3,1] + zeta[21,3,1] <= 0
 R336: - beta[3,2] + zeta[21,3,2] <= 0
 R337: - beta[3,1] + zeta[49,3,1] <= 0
 R338: - beta[3,2] + zeta[49,3,2] <= 0
 R339: - beta[3,1] + zeta[48,3,1] <= 0
 R340: - beta[3,2] + zeta[48,3,2] <= 0
 R341: - beta[3,1] + zeta[34,3,1] <= 0
 R342: - beta[3,2] + zeta[34,3,2] <= 0
 R343: - beta[3,1] + zeta[9,3,1] <= 0
 R344: - beta[3,2] + zeta[9,3,2] <= 0
 R345: - beta[3,1] + zeta[45,3,1] <= 0
 R346: - beta[3,2] + zeta[45,3,2] <= 0
 R347: - beta[3,1] + zeta[23,3,1] <= 0
 R348: - beta[3,2] + zeta[23,3,2] <= 0
 R349: - beta[3,1] + zeta[19,3,1] <= 0
 R350: - beta[3,2] + zeta[19,3,2] <= 0
 R351: - beta[3,1] + zeta[6,3,1] <= 0
 R352: - beta[3,2] + zeta[6,3,2] <= 0
 R353: - beta[3,1] + zeta[25,3,1] <= 0
 R354: - beta[3,2] + zeta[25,3,2] <= 0
 R355: - beta[3,1] + zeta[4,3,1] <= 0
 R356: - beta[3,2] + zeta[4,3,2] <= 0
 R357: - beta[3,1] + zeta[26,3,1] <= 0
 R358: - beta[3,2] + zeta[26,3,2] <= 0
 R359: - beta[3,1] + zeta[41,3,1] <= 0
 R360: - beta[3,2] + zeta[41,3,2] <= 0
 R361: - beta[3,1] + zeta[31,3,1] <= 0
 R362: - beta[3,2] + zeta[31,3,2] <= 0
 R363: - beta[3,1] + zeta[7,3,1] <= 0
 R364: - beta[3,2] + zeta[7,3,2] <= 0
 R365: - beta[3,1] + zeta[0,3,1] <= 0
 R366: - beta[3,2] + zeta[0,3,2] <= 0
 R367: - beta[4,1] + zeta[43,4,1] <= 0
 R368: - beta[4,2] + zeta[43,4,2] <= 0
 R369: - beta[4,1] + zeta[1,4,1] <= 0
 R370: - beta[4,2] + zeta[1,4,2] <= 0
 R371: - beta[4,1] + zeta[46,4,1] <= 0
 R372: - beta[4,2] + zeta[46,4,2] <= 0
 R373: - beta[4,1] + zeta[38,4,1] <= 0
 R374: - beta[4,2] + zeta[38,4,2] <= 0
 R375: - beta[4,1] + zeta[16,4,1] <= 0
 R376: - beta[4,2] + zeta[16,4,2] <= 0
 R377: - beta[4,1] + zeta[27,4,1] <= 0
 R378: - beta[4,2] + zeta[27,4,2] <= 0
 R379: - beta[4,1] + zeta[3,4,1] <= 0
 R380: - beta[4,2] + zeta[3,4,2] <= 0
 R381: - beta[4,1] + zeta[24,4,1] <= 0
 R382: - beta[4,2] + zeta[24,4,2] <= 0
 R383: - beta[4,1] + zeta[21,4,1] <= 0
 R384: - beta[4,2] + zeta[21,4,2] <= 0
 R385: - beta[4,1] + zeta[49,4,1] <= 0
 R386: - beta[4,2] + zeta[49,4,2] <= 0
 R387: - beta[4,1] + zeta[48,4,1] <= 0
 R388: - beta[4,2] + zeta[48,4,2] <= 0
 R389: - beta[4,1] + zeta[34,4,1] <= 0
 R390: - beta[4,2] + zeta[34,4,2] <= 0
 R391: - beta[4,1] + zeta[9,4,1] <= 0
 R392: - beta[4,2] + zeta[9,4,2] <= 0
 R393: - beta[4,1] + zeta[45,4,1] <= 0
 R394: - beta[4,2] + zeta[45,4,2] <= 0
 R395: - beta[4,1] + zeta[23,4,1] <= 0
 R396: - beta[4,2] + zeta[23,4,2] <= 0
 R397: - beta[4,1] + zeta[19,4,1] <= 0
 R398: - beta[4,2] + zeta[19,4,2] <= 0
 R399: - beta[4,1] + zeta[6,4,1] <= 0
 R400: - beta[4,2] + zeta[6,4,2] <= 0
 R401: - beta[4,1] + zeta[25,4,1] <= 0
 R402: - beta[4,2] + zeta[25,4,2] <= 0
 R403: - beta[4,1] + zeta[4,4,1] <= 0
 R404: - beta[4,2] + zeta[4,4,2] <= 0
 R405: - beta[4,1] + zeta[26,4,1] <= 0
 R406: - beta[4,2] + zeta[26,4,2] <= 0
 R407: - beta[4,1] + zeta[41,4,1] <= 0
 R408: - beta[4,2] + zeta[41,4,2] <= 0
 R409: - beta[4,1] + zeta[31,4,1] <= 0
 R410: - beta[4,2] + zeta[31,4,2] <= 0
 R411: - beta[4,1] + zeta[7,4,1] <= 0
 R412: - beta[4,2] + zeta[7,4,2] <= 0
 R413: - beta[4,1] + zeta[0,4,1] <= 0
 R414: - beta[4,2] + zeta[0,4,2] <= 0
 R415: - beta[5,1] + zeta[43,5,1] <= 0
 R416: - beta[5,2] + zeta[43,5,2] <= 0
 R417: - beta[5,1] + zeta[1,5,1] <= 0
 R418: - beta[5,2] + zeta[1,5,2] <= 0
 R419: - beta[5,1] + zeta[46,5,1] <= 0
 R420: - beta[5,2] + zeta[46,5,2] <= 0
 R421: - beta[5,1] + zeta[38,5,1] <= 0
 R422: - beta[5,2] + zeta[38,5,2] <= 0
 R423: - beta[5,1] + zeta[16,5,1] <= 0
 R424: - beta[5,2] + zeta[16,5,2] <= 0
 R425: - beta[5,1] + zeta[27,5,1] <= 0
 R426: - beta[5,2] + zeta[27,5,2] <= 0
 R427: - beta[5,1] + zeta[3,5,1] <= 0
 R428: - beta[5,2] + zeta[3,5,2] <= 0
 R429: - beta[5,1] + zeta[24,5,1] <= 0
 R430: - beta[5,2] + zeta[24,5,2] <= 0
 R431: - beta[5,1] + zeta[21,5,1] <= 0
 R432: - beta[5,2] + zeta[21,5,2] <= 0
 R433: - beta[5,1] + zeta[49,5,1] <= 0
 R434: - beta[5,2] + zeta[49,5,2] <= 0
 R435: - beta[5,1] + zeta[48,5,1] <= 0
 R436: - beta[5,2] + zeta[48,5,2] <= 0
 R437: - beta[5,1] + zeta[34,5,1] <= 0
 R438: - beta[5,2] + zeta[34,5,2] <= 0
 R439: - beta[5,1] + zeta[9,5,1] <= 0
 R440: - beta[5,2] + zeta[9,5,2] <= 0
 R441: - beta[5,1] + zeta[45,5,1] <= 0
 R442: - beta[5,2] + zeta[45,5,2] <= 0
 R443: - beta[5,1] + zeta[23,5,1] <= 0
 R444: - beta[5,2] + zeta[23,5,2] <= 0
 R445: - beta[5,1] + zeta[19,5,1] <= 0
 R446: - beta[5,2] + zeta[19,5,2] <= 0
 R447: - beta[5,1] + zeta[6,5,1] <= 0
 R448: - beta[5,2] + zeta[6,5,2] <= 0
 R449: - beta[5,1] + zeta[25,5,1] <= 0
 R450: - beta[5,2] + zeta[25,5,2] <= 0
 R451: - beta[5,1] + zeta[4,5,1] <= 0
 R452: - beta[5,2] + zeta[4,5,2] <= 0
 R453: - beta[5,1] + zeta[26,5,1] <= 0
 R454: - beta[5,2] + zeta[26,5,2] <= 0
 R455: - beta[5,1] + zeta[41,5,1] <= 0
 R456: - beta[5,2] + zeta[41,5,2] <= 0
 R457: - beta[5,1] + zeta[31,5,1] <= 0
 R458: - beta[5,2] + zeta[31,5,2] <= 0
 R459: - beta[5,1] + zeta[7,5,1] <= 0
 R460: - beta[5,2] + zeta[7,5,2] <= 0
 R461: - beta[5,1] + zeta[0,5,1] <= 0
 R462: - beta[5,2] + zeta[0,5,2] <= 0
 R463: - beta[6,1] + zeta[43,6,1] <= 0
 R464: - beta[6,2] + zeta[43,6,2] <= 0
 R465: - beta[6,1] + zeta[1,6,1] <= 0
 R466: - beta[6,2] + zeta[1,6,2] <= 0
 R467: - beta[6,1] + zeta[46,6,1] <= 0
 R468: - beta[6,2] + zeta[46,6,2] <= 0
 R469: - beta[6,1] + zeta[38,6,1] <= 0
 R470: - beta[6,2] + zeta[38,6,2] <= 0
 R471: - beta[6,1] + zeta[16,6,1] <= 0
 R472: - beta[6,2] + zeta[16,6,2] <= 0
 R473: - beta[6,1] + zeta[27,6,1] <= 0
 R474: - beta[6,2] + zeta[27,6,2] <= 0
 R475: - beta[6,1] + zeta[3,6,1] <= 0
 R476: - beta[6,2] + zeta[3,6,2] <= 0
 R477: - beta[6,1] + zeta[24,6,1] <= 0
 R478: - beta[6,2] + zeta[24,6,2] <= 0
 R479: - beta[6,1] + zeta[21,6,1] <= 0
 R480: - beta[6,2] + zeta[21,6,2] <= 0
 R481: - beta[6,1] + zeta[49,6,1] <= 0
 R482: - beta[6,2] + zeta[49,6,2] <= 0
 R483: - beta[6,1] + zeta[48,6,1] <= 0
 R484: - beta[6,2] + zeta[48,6,2] <= 0
 R485: - beta[6,1] + zeta[34,6,1] <= 0
 R486: - beta[6,2] + zeta[34,6,2] <= 0
 R487: - beta[6,1] + zeta[9,6,1] <= 0
 R488: - beta[6,2] + zeta[9,6,2] <= 0
 R489: - beta[6,1] + zeta[45,6,1] <= 0
 R490: - beta[6,2] + zeta[45,6,2] <= 0
 R491: - beta[6,1] + zeta[23,6,1] <= 0
 R492: - beta[6,2] + zeta[23,6,2] <= 0
 R493: - beta[6,1] + zeta[19,6,1] <= 0
 R494: - beta[6,2] + zeta[19,6,2] <= 0
 R495: - beta[6,1] + zeta[6,6,1] <= 0
 R496: - beta[6,2] + zeta[6,6,2] <= 0
 R497: - beta[6,1] + zeta[25,6,1] <= 0
 R498: - beta[6,2] + zeta[25,6,2] <= 0
 R499: - beta[6,1] + zeta[4,6,1] <= 0
 R500: - beta[6,2] + zeta[4,6,2] <= 0
 R501: - beta[6,1] + zeta[26,6,1] <= 0
 R502: - beta[6,2] + zeta[26,6,2] <= 0
 R503: - beta[6,1] + zeta[41,6,1] <= 0
 R504: - beta[6,2] + zeta[41,6,2] <= 0
 R505: - beta[6,1] + zeta[31,6,1] <= 0
 R506: - beta[6,2] + zeta[31,6,2] <= 0
 R507: - beta[6,1] + zeta[7,6,1] <= 0
 R508: - beta[6,2] + zeta[7,6,2] <= 0
 R509: - beta[6,1] + zeta[0,6,1] <= 0
 R510: - beta[6,2] + zeta[0,6,2] <= 0
 R511: - beta[7,1] + zeta[43,7,1] <= 0
 R512: - beta[7,2] + zeta[43,7,2] <= 0
 R513: - beta[7,1] + zeta[1,7,1] <= 0
 R514: - beta[7,2] + zeta[1,7,2] <= 0
 R515: - beta[7,1] + zeta[46,7,1] <= 0
 R516: - beta[7,2] + zeta[46,7,2] <= 0
 R517: - beta[7,1] + zeta[38,7,1] <= 0
 R518: - beta[7,2] + zeta[38,7,2] <= 0
 R519: - beta[7,1] + zeta[16,7,1] <= 0
 R520: - beta[7,2] + zeta[16,7,2] <= 0
 R521: - beta[7,1] + zeta[27,7,1] <= 0
 R522: - beta[7,2] + zeta[27,7,2] <= 0
 R523: - beta[7,1] + zeta[3,7,1] <= 0
 R524: - beta[7,2] + zeta[3,7,2] <= 0
 R525: - beta[7,1] + zeta[24,7,1] <= 0
 R526: - beta[7,2] + zeta[24,7,2] <= 0
 R527: - beta[7,1] + zeta[21,7,1] <= 0
 R528: - beta[7,2] + zeta[21,7,2] <= 0
 R529: - beta[7,1] + zeta[49,7,1] <= 0
 R530: - beta[7,2] + zeta[49,7,2] <= 0
 R531: - beta[7,1] + zeta[48,7,1] <= 0
 R532: - beta[7,2] + zeta[48,7,2] <= 0
 R533: - beta[7,1] + zeta[34,7,1] <= 0
 R534: - beta[7,2] + zeta[34,7,2] <= 0
 R535: - beta[7,1] + zeta[9,7,1] <= 0
 R536: - beta[7,2] + zeta[9,7,2] <= 0
 R537: - beta[7,1] + zeta[45,7,1] <= 0
 R538: - beta[7,2] + zeta[45,7,2] <= 0
 R539: - beta[7,1] + zeta[23,7,1] <= 0
 R540: - beta[7,2] + zeta[23,7,2] <= 0
 R541: - beta[7,1] + zeta[19,7,1] <= 0
 R542: - beta[7,2] + zeta[19,7,2] <= 0
 R543: - beta[7,1] + zeta[6,7,1] <= 0
 R544: - beta[7,2] + zeta[6,7,2] <= 0
 R545: - beta[7,1] + zeta[25,7,1] <= 0
 R546: - beta[7,2] + zeta[25,7,2] <= 0
 R547: - beta[7,1] + zeta[4,7,1] <= 0
 R548: - beta[7,2] + zeta[4,7,2] <= 0
 R549: - beta[7,1] + zeta[26,7,1] <= 0
 R550: - beta[7,2] + zeta[26,7,2] <= 0
 R551: - beta[7,1] + zeta[41,7,1] <= 0
 R552: - beta[7,2] + zeta[41,7,2] <= 0
 R553: - beta[7,1] + zeta[31,7,1] <= 0
 R554: - beta[7,2] + zeta[31,7,2] <= 0
 R555: - beta[7,1] + zeta[7,7,1] <= 0
 R556: - beta[7,2] + zeta[7,7,2] <= 0
 R557: - beta[7,1] + zeta[0,7,1] <= 0
 R558: - beta[7,2] + zeta[0,7,2] <= 0
 R559: - p[1] + beta[1,1] + beta[1,2] = 0
 R560: - p[2] + beta[2,1] + beta[2,2] = 0
 R561: - p[3] + beta[3,1] + beta[3,2] = 0
 R562: - p[4] + beta[4,1] + beta[4,2] = 0
 R563: - p[5] + beta[5,1] + beta[5,2] = 0
 R564: - p[6] + beta[6,1] + beta[6,2] = 0
 R565: - p[7] + beta[7,1] + beta[7,2] = 0
 R566: zeta[43,4,1] + zeta[43,4,2] - z[43,4] = 0
 R567: zeta[1,4,1] + zeta[1,4,2] - z[1,4] = 0
 R568: zeta[46,4,1] + zeta[46,4,2] - z[46,4] = 0
 R569: zeta[38,4,1] + zeta[38,4,2] - z[38,4] = 0
 R570: zeta[16,4,1] + zeta[16,4,2] - z[16,4] = 0
 R571: zeta[27,4,1] + zeta[27,4,2] - z[27,4] = 0
 R572: zeta[3,4,1] + zeta[3,4,2] - z[3,4] = 0
 R573: zeta[24,4,1] + zeta[24,4,2] - z[24,4] = 0
 R574: zeta[21,4,1] + zeta[21,4,2] - z[21,4] = 0
 R575: zeta[49,4,1] + zeta[49,4,2] - z[49,4] = 0
 R576: zeta[48,4,1] + zeta[48,4,2] - z[48,4] = 0
 R577: zeta[34,4,1] + zeta[34,4,2] - z[34,4] = 0
 R578: zeta[9,4,1] + zeta[9,4,2] - z[9,4] = 0
 R579: zeta[45,4,1] + zeta[45,4,2] - z[45,4] = 0
 R580: zeta[23,4,1] + zeta[23,4,2] - z[23,4] = 0
 R581: zeta[19,4,1] + zeta[19,4,2] - z[19,4] = 0
 R582: zeta[6,4,1] + zeta[6,4,2] - z[6,4] = 0
 R583: zeta[25,4,1] + zeta[25,4,2] - z[25,4] = 0
 R584: zeta[4,4,1] + zeta[4,4,2] - z[4,4] = 0
 R585: zeta[26,4,1] + zeta[26,4,2] - z[26,4] = 0
 R586: zeta[41,4,1] + zeta[41,4,2] - z[41,4] = 0
 R587: zeta[31,4,1] + zeta[31,4,2] - z[31,4] = 0
 R588: zeta[7,4,1] + zeta[7,4,2] - z[7,4] = 0
 R589: zeta[0,4,1] + zeta[0,4,2] - z[0,4] = 0
 R590: zeta[43,5,1] + zeta[43,5,2] - z[43,5] = 0
 R591: zeta[1,5,1] + zeta[1,5,2] - z[1,5] = 0
 R592: zeta[46,5,1] + zeta[46,5,2] - z[46,5] = 0
 R593: zeta[38,5,1] + zeta[38,5,2] - z[38,5] = 0
 R594: zeta[16,5,1] + zeta[16,5,2] - z[16,5] = 0
 R595: zeta[27,5,1] + zeta[27,5,2] - z[27,5] = 0
 R596: zeta[3,5,1] + zeta[3,5,2] - z[3,5] = 0
 R597: zeta[24,5,1] + zeta[24,5,2] - z[24,5] = 0
 R598: zeta[21,5,1] + zeta[21,5,2] - z[21,5] = 0
 R599: zeta[49,5,1] + zeta[49,5,2] - z[49,5] = 0
 R600: zeta[48,5,1] + zeta[48,5,2] - z[48,5] = 0
 R601: zeta[34,5,1] + zeta[34,5,2] - z[34,5] = 0
 R602: zeta[9,5,1] + zeta[9,5,2] - z[9,5] = 0
 R603: zeta[45,5,1] + zeta[45,5,2] - z[45,5] = 0
 R604: zeta[23,5,1] + zeta[23,5,2] - z[23,5] = 0
 R605: zeta[19,5,1] + zeta[19,5,2] - z[19,5] = 0
 R606: zeta[6,5,1] + zeta[6,5,2] - z[6,5] = 0
 R607: zeta[25,5,1] + zeta[25,5,2] - z[25,5] = 0
 R608: zeta[4,5,1] + zeta[4,5,2] - z[4,5] = 0
 R609: zeta[26,5,1] + zeta[26,5,2] - z[26,5] = 0
 R610: zeta[41,5,1] + zeta[41,5,2] - z[41,5] = 0
 R611: zeta[31,5,1] + zeta[31,5,2] - z[31,5] = 0
 R612: zeta[7,5,1] + zeta[7,5,2] - z[7,5] = 0
 R613: zeta[0,5,1] + zeta[0,5,2] - z[0,5] = 0
 R614: zeta[43,6,1] + zeta[43,6,2] - z[43,6] = 0
 R615: zeta[1,6,1] + zeta[1,6,2] - z[1,6] = 0
 R616: zeta[46,6,1] + zeta[46,6,2] - z[46,6] = 0
 R617: zeta[38,6,1] + zeta[38,6,2] - z[38,6] = 0
 R618: zeta[16,6,1] + zeta[16,6,2] - z[16,6] = 0
 R619: zeta[27,6,1] + zeta[27,6,2] - z[27,6] = 0
 R620: zeta[3,6,1] + zeta[3,6,2] - z[3,6] = 0
 R621: zeta[24,6,1] + zeta[24,6,2] - z[24,6] = 0
 R622: zeta[21,6,1] + zeta[21,6,2] - z[21,6] = 0
 R623: zeta[49,6,1] + zeta[49,6,2] - z[49,6] = 0
 R624: zeta[48,6,1] + zeta[48,6,2] - z[48,6] = 0
 R625: zeta[34,6,1] + zeta[34,6,2] - z[34,6] = 0
 R626: zeta[9,6,1] + zeta[9,6,2] - z[9,6] = 0
 R627: zeta[45,6,1] + zeta[45,6,2] - z[45,6] = 0
 R628: zeta[23,6,1] + zeta[23,6,2] - z[23,6] = 0
 R629: zeta[19,6,1] + zeta[19,6,2] - z[19,6] = 0
 R630: zeta[6,6,1] + zeta[6,6,2] - z[6,6] = 0
 R631: zeta[25,6,1] + zeta[25,6,2] - z[25,6] = 0
 R632: zeta[4,6,1] + zeta[4,6,2] - z[4,6] = 0
 R633: zeta[26,6,1] + zeta[26,6,2] - z[26,6] = 0
 R634: zeta[41,6,1] + zeta[41,6,2] - z[41,6] = 0
 R635: zeta[31,6,1] + zeta[31,6,2] - z[31,6] = 0
 R636: zeta[7,6,1] + zeta[7,6,2] - z[7,6] = 0
 R637: zeta[0,6,1] + zeta[0,6,2] - z[0,6] = 0
 R638: zeta[43,7,1] + zeta[43,7,2] - z[43,7] = 0
 R639: zeta[1,7,1] + zeta[1,7,2] - z[1,7] = 0
 R640: zeta[46,7,1] + zeta[46,7,2] - z[46,7] = 0
 R641: zeta[38,7,1] + zeta[38,7,2] - z[38,7] = 0
 R642: zeta[16,7,1] + zeta[16,7,2] - z[16,7] = 0
 R643: zeta[27,7,1] + zeta[27,7,2] - z[27,7] = 0
 R644: zeta[3,7,1] + zeta[3,7,2] - z[3,7] = 0
 R645: zeta[24,7,1] + zeta[24,7,2] - z[24,7] = 0
 R646: zeta[21,7,1] + zeta[21,7,2] - z[21,7] = 0
 R647: zeta[49,7,1] + zeta[49,7,2] - z[49,7] = 0
 R648: zeta[48,7,1] + zeta[48,7,2] - z[48,7] = 0
 R649: zeta[34,7,1] + zeta[34,7,2] - z[34,7] = 0
 R650: zeta[9,7,1] + zeta[9,7,2] - z[9,7] = 0
 R651: zeta[45,7,1] + zeta[45,7,2] - z[45,7] = 0
 R652: zeta[23,7,1] + zeta[23,7,2] - z[23,7] = 0
 R653: zeta[19,7,1] + zeta[19,7,2] - z[19,7] = 0
 R654: zeta[6,7,1] + zeta[6,7,2] - z[6,7] = 0
 R655: zeta[25,7,1] + zeta[25,7,2] - z[25,7] = 0
 R656: zeta[4,7,1] + zeta[4,7,2] - z[4,7] = 0
 R657: zeta[26,7,1] + zeta[26,7,2] - z[26,7] = 0
 R658: zeta[41,7,1] + zeta[41,7,2] - z[41,7] = 0
 R659: zeta[31,7,1] + zeta[31,7,2] - z[31,7] = 0
 R660: zeta[7,7,1] + zeta[7,7,2] - z[7,7] = 0
 R661: zeta[0,7,1] + zeta[0,7,2] - z[0,7] = 0
 R662: 0.1111111111111111 zeta[43,1,2] + 0.1111111111111111 zeta[43,2,2]
   + 0.1111111111111111 zeta[43,3,2] + 0.1111111111111111 zeta[43,4,2]
   + 0.1111111111111111 zeta[43,5,2] + 0.1111111111111111 zeta[43,6,2]
   + 0.1111111111111111 zeta[43,7,2] + 0.1111111111111111 zeta[1,1,2]
   + 0.1111111111111111 zeta[1,2,2] + 0.1111111111111111 zeta[1,3,2]
   + 0.1111111111111111 zeta[1,4,2] + 0.1111111111111111 zeta[1,5,2]
   + 0.1111111111111111 zeta[1,6,2] + 0.1111111111111111 zeta[1,7,2]
   + 0.1111111111111111 zeta[46,1,2] + 0.1111111111111111 zeta[46,2,2]
   + 0.1111111111111111 zeta[46,3,2] + 0.1111111111111111 zeta[46,4,2]
   + 0.1111111111111111 zeta[46,5,2] + 0.1111111111111111 zeta[46,6,2]
   + 0.1111111111111111 zeta[46,7,2] + 0.1111111111111111 zeta[38,1,2]
   + 0.1111111111111111 zeta[38,2,2] + 0.1111111111111111 zeta[38,3,2]
   + 0.1111111111111111 zeta[38,4,2] + 0.1111111111111111 zeta[38,5,2]
   + 0.1111111111111111 zeta[38,6,2] + 0.1111111111111111 zeta[38,7,2]
   - 0.1111111111111111 zeta[16,1,2] - 0.1111111111111111 zeta[16,2,2]
   - 0.1111111111111111 zeta[16,3,2] - 0.1111111111111111 zeta[16,4,2]
   - 0.1111111111111111 zeta[16,5,2] - 0.1111111111111111 zeta[16,6,2]
   - 0.1111111111111111 zeta[16,7,2] - 0.1111111111111111 zeta[27,1,2]
   - 0.1111111111111111 zeta[27,2,2] - 0.1111111111111111 zeta[27,3,2]
   - 0.1111111111111111 zeta[27,4,2] - 0.1111111111111111 zeta[27,5,2]
   - 0.1111111111111111 zeta[27,6,2] - 0.1111111111111111 zeta[27,7,2]
   + 0.1111111111111111 zeta[24,1,2] + 0.1111111111111111 zeta[24,2,2]
   + 0.1111111111111111 zeta[24,3,2] + 0.1111111111111111 zeta[24,4,2]
   + 0.1111111111111111 zeta[24,5,2] + 0.1111111111111111 zeta[24,6,2]
   + 0.1111111111111111 zeta[24,7,2] + 0.1111111111111111 zeta[49,1,2]
   + 0.1111111111111111 zeta[49,2,2] + 0.1111111111111111 zeta[49,3,2]
   + 0.1111111111111111 zeta[49,4,2] + 0.1111111111111111 zeta[49,5,2]
   + 0.1111111111111111 zeta[49,6,2] + 0.1111111111111111 zeta[49,7,2]
   - 0.1111111111111111 zeta[34,1,2] - 0.1111111111111111 zeta[34,2,2]
   - 0.1111111111111111 zeta[34,3,2] - 0.1111111111111111 zeta[34,4,2]
   - 0.1111111111111111 zeta[34,5,2] - 0.1111111111111111 zeta[34,6,2]
   - 0.1111111111111111 zeta[34,7,2] - 0.1111111111111111 zeta[9,1,2]
   - 0.1111111111111111 zeta[9,2,2] - 0.1111111111111111 zeta[9,3,2]
   - 0.1111111111111111 zeta[9,4,2] - 0.1111111111111111 zeta[9,5,2]
   - 0.1111111111111111 zeta[9,6,2] - 0.1111111111111111 zeta[9,7,2]
   - 0.1111111111111111 zeta[19,1,2] - 0.1111111111111111 zeta[19,2,2]
   - 0.1111111111111111 zeta[19,3,2] - 0.1111111111111111 zeta[19,4,2]
   - 0.1111111111111111 zeta[19,5,2] - 0.1111111111111111 zeta[19,6,2]
   - 0.1111111111111111 zeta[19,7,2] - 0.1111111111111111 zeta[6,1,2]
   - 0.1111111111111111 zeta[6,2,2] - 0.1111111111111111 zeta[6,3,2]
   - 0.1111111111111111 zeta[6,4,2] - 0.1111111111111111 zeta[6,5,2]
   - 0.1111111111111111 zeta[6,6,2] - 0.1111111111111111 zeta[6,7,2]
   + 0.1111111111111111 zeta[25,1,2] + 0.1111111111111111 zeta[25,2,2]
   + 0.1111111111111111 zeta[25,3,2] + 0.1111111111111111 zeta[25,4,2]
   + 0.1111111111111111 zeta[25,5,2] + 0.1111111111111111 zeta[25,6,2]
   + 0.1111111111111111 zeta[25,7,2] - 0.1111111111111111 zeta[4,1,2]
   - 0.1111111111111111 zeta[4,2,2] - 0.1111111111111111 zeta[4,3,2]
   - 0.1111111111111111 zeta[4,4,2] - 0.1111111111111111 zeta[4,5,2]
   - 0.1111111111111111 zeta[4,6,2] - 0.1111111111111111 zeta[4,7,2]
   + 0.1111111111111111 zeta[26,1,2] + 0.1111111111111111 zeta[26,2,2]
   + 0.1111111111111111 zeta[26,3,2] + 0.1111111111111111 zeta[26,4,2]
   + 0.1111111111111111 zeta[26,5,2] + 0.1111111111111111 zeta[26,6,2]
   + 0.1111111111111111 zeta[26,7,2] + 0.1111111111111111 zeta[41,1,2]
   + 0.1111111111111111 zeta[41,2,2] + 0.1111111111111111 zeta[41,3,2]
   + 0.1111111111111111 zeta[41,4,2] + 0.1111111111111111 zeta[41,5,2]
   + 0.1111111111111111 zeta[41,6,2] + 0.1111111111111111 zeta[41,7,2]
   - 0.1111111111111111 zeta[31,1,2] - 0.1111111111111111 zeta[31,2,2]
   - 0.1111111111111111 zeta[31,3,2] - 0.1111111111111111 zeta[31,4,2]
   - 0.1111111111111111 zeta[31,5,2] - 0.1111111111111111 zeta[31,6,2]
   - 0.1111111111111111 zeta[31,7,2] - 0.1111111111111111 zeta[7,1,2]
   - 0.1111111111111111 zeta[7,2,2] - 0.1111111111111111 zeta[7,3,2]
   - 0.1111111111111111 zeta[7,4,2] - 0.1111111111111111 zeta[7,5,2]
   - 0.1111111111111111 zeta[7,6,2] - 0.1111111111111111 zeta[7,7,2]
   <= 0.01
 R663: 0.1111111111111111 zeta[43,1,2] + 0.1111111111111111 zeta[43,2,2]
   + 0.1111111111111111 zeta[43,3,2] + 0.1111111111111111 zeta[43,4,2]
   + 0.1111111111111111 zeta[43,5,2] + 0.1111111111111111 zeta[43,6,2]
   + 0.1111111111111111 zeta[43,7,2] + 0.1111111111111111 zeta[1,1,2]
   + 0.1111111111111111 zeta[1,2,2] + 0.1111111111111111 zeta[1,3,2]
   + 0.1111111111111111 zeta[1,4,2] + 0.1111111111111111 zeta[1,5,2]
   + 0.1111111111111111 zeta[1,6,2] + 0.1111111111111111 zeta[1,7,2]
   + 0.1111111111111111 zeta[46,1,2] + 0.1111111111111111 zeta[46,2,2]
   + 0.1111111111111111 zeta[46,3,2] + 0.1111111111111111 zeta[46,4,2]
   + 0.1111111111111111 zeta[46,5,2] + 0.1111111111111111 zeta[46,6,2]
   + 0.1111111111111111 zeta[46,7,2] + 0.1111111111111111 zeta[38,1,2]
   + 0.1111111111111111 zeta[38,2,2] + 0.1111111111111111 zeta[38,3,2]
   + 0.1111111111111111 zeta[38,4,2] + 0.1111111111111111 zeta[38,5,2]
   + 0.1111111111111111 zeta[38,6,2] + 0.1111111111111111 zeta[38,7,2]
   - 0.1111111111111111 zeta[16,1,2] - 0.1111111111111111 zeta[16,2,2]
   - 0.1111111111111111 zeta[16,3,2] - 0.1111111111111111 zeta[16,4,2]
   - 0.1111111111111111 zeta[16,5,2] - 0.1111111111111111 zeta[16,6,2]
   - 0.1111111111111111 zeta[16,7,2] - 0.1111111111111111 zeta[27,1,2]
   - 0.1111111111111111 zeta[27,2,2] - 0.1111111111111111 zeta[27,3,2]
   - 0.1111111111111111 zeta[27,4,2] - 0.1111111111111111 zeta[27,5,2]
   - 0.1111111111111111 zeta[27,6,2] - 0.1111111111111111 zeta[27,7,2]
   + 0.1111111111111111 zeta[24,1,2] + 0.1111111111111111 zeta[24,2,2]
   + 0.1111111111111111 zeta[24,3,2] + 0.1111111111111111 zeta[24,4,2]
   + 0.1111111111111111 zeta[24,5,2] + 0.1111111111111111 zeta[24,6,2]
   + 0.1111111111111111 zeta[24,7,2] + 0.1111111111111111 zeta[49,1,2]
   + 0.1111111111111111 zeta[49,2,2] + 0.1111111111111111 zeta[49,3,2]
   + 0.1111111111111111 zeta[49,4,2] + 0.1111111111111111 zeta[49,5,2]
   + 0.1111111111111111 zeta[49,6,2] + 0.1111111111111111 zeta[49,7,2]
   - 0.1111111111111111 zeta[34,1,2] - 0.1111111111111111 zeta[34,2,2]
   - 0.1111111111111111 zeta[34,3,2] - 0.1111111111111111 zeta[34,4,2]
   - 0.1111111111111111 zeta[34,5,2] - 0.1111111111111111 zeta[34,6,2]
   - 0.1111111111111111 zeta[34,7,2] - 0.1111111111111111 zeta[9,1,2]
   - 0.1111111111111111 zeta[9,2,2] - 0.1111111111111111 zeta[9,3,2]
   - 0.1111111111111111 zeta[9,4,2] - 0.1111111111111111 zeta[9,5,2]
   - 0.1111111111111111 zeta[9,6,2] - 0.1111111111111111 zeta[9,7,2]
   - 0.1111111111111111 zeta[19,1,2] - 0.1111111111111111 zeta[19,2,2]
   - 0.1111111111111111 zeta[19,3,2] - 0.1111111111111111 zeta[19,4,2]
   - 0.1111111111111111 zeta[19,5,2] - 0.1111111111111111 zeta[19,6,2]
   - 0.1111111111111111 zeta[19,7,2] - 0.1111111111111111 zeta[6,1,2]
   - 0.1111111111111111 zeta[6,2,2] - 0.1111111111111111 zeta[6,3,2]
   - 0.1111111111111111 zeta[6,4,2] - 0.1111111111111111 zeta[6,5,2]
   - 0.1111111111111111 zeta[6,6,2] - 0.1111111111111111 zeta[6,7,2]
   + 0.1111111111111111 zeta[25,1,2] + 0.1111111111111111 zeta[25,2,2]
   + 0.1111111111111111 zeta[25,3,2] + 0.1111111111111111 zeta[25,4,2]
   + 0.1111111111111111 zeta[25,5,2] + 0.1111111111111111 zeta[25,6,2]
   + 0.1111111111111111 zeta[25,7,2] - 0.1111111111111111 zeta[4,1,2]
   - 0.1111111111111111 zeta[4,2,2] - 0.1111111111111111 zeta[4,3,2]
   - 0.1111111111111111 zeta[4,4,2] - 0.1111111111111111 zeta[4,5,2]
   - 0.1111111111111111 zeta[4,6,2] - 0.1111111111111111 zeta[4,7,2]
   + 0.1111111111111111 zeta[26,1,2] + 0.1111111111111111 zeta[26,2,2]
   + 0.1111111111111111 zeta[26,3,2] + 0.1111111111111111 zeta[26,4,2]
   + 0.1111111111111111 zeta[26,5,2] + 0.1111111111111111 zeta[26,6,2]
   + 0.1111111111111111 zeta[26,7,2] + 0.1111111111111111 zeta[41,1,2]
   + 0.1111111111111111 zeta[41,2,2] + 0.1111111111111111 zeta[41,3,2]
   + 0.1111111111111111 zeta[41,4,2] + 0.1111111111111111 zeta[41,5,2]
   + 0.1111111111111111 zeta[41,6,2] + 0.1111111111111111 zeta[41,7,2]
   - 0.1111111111111111 zeta[31,1,2] - 0.1111111111111111 zeta[31,2,2]
   - 0.1111111111111111 zeta[31,3,2] - 0.1111111111111111 zeta[31,4,2]
   - 0.1111111111111111 zeta[31,5,2] - 0.1111111111111111 zeta[31,6,2]
   - 0.1111111111111111 zeta[31,7,2] - 0.1111111111111111 zeta[7,1,2]
   - 0.1111111111111111 zeta[7,2,2] - 0.1111111111111111 zeta[7,3,2]
   - 0.1111111111111111 zeta[7,4,2] - 0.1111111111111111 zeta[7,5,2]
   - 0.1111111111111111 zeta[7,6,2] - 0.1111111111111111 zeta[7,7,2]
   >= -0.01
 R664: 0.1111111111111111 zeta[43,1,2] + 0.1111111111111111 zeta[43,2,2]
   + 0.1111111111111111 zeta[43,3,2] + 0.1111111111111111 zeta[43,4,2]
   + 0.1111111111111111 zeta[43,5,2] + 0.1111111111111111 zeta[43,6,2]
   + 0.1111111111111111 zeta[43,7,2] + 0.1111111111111111 zeta[1,1,2]
   + 0.1111111111111111 zeta[1,2,2] + 0.1111111111111111 zeta[1,3,2]
   + 0.1111111111111111 zeta[1,4,2] + 0.1111111111111111 zeta[1,5,2]
   + 0.1111111111111111 zeta[1,6,2] + 0.1111111111111111 zeta[1,7,2]
   + 0.1111111111111111 zeta[46,1,2] + 0.1111111111111111 zeta[46,2,2]
   + 0.1111111111111111 zeta[46,3,2] + 0.1111111111111111 zeta[46,4,2]
   + 0.1111111111111111 zeta[46,5,2] + 0.1111111111111111 zeta[46,6,2]
   + 0.1111111111111111 zeta[46,7,2] + 0.1111111111111111 zeta[38,1,2]
   + 0.1111111111111111 zeta[38,2,2] + 0.1111111111111111 zeta[38,3,2]
   + 0.1111111111111111 zeta[38,4,2] + 0.1111111111111111 zeta[38,5,2]
   + 0.1111111111111111 zeta[38,6,2] + 0.1111111111111111 zeta[38,7,2]
   - 0.25 zeta[3,1,2] - 0.25 zeta[3,2,2] - 0.25 zeta[3,3,2]
   - 0.25 zeta[3,4,2] - 0.25 zeta[3,5,2] - 0.25 zeta[3,6,2]
   - 0.25 zeta[3,7,2] + 0.1111111111111111 zeta[24,1,2]
   + 0.1111111111111111 zeta[24,2,2] + 0.1111111111111111 zeta[24,3,2]
   + 0.1111111111111111 zeta[24,4,2] + 0.1111111111111111 zeta[24,5,2]
   + 0.1111111111111111 zeta[24,6,2] + 0.1111111111111111 zeta[24,7,2]
   + 0.1111111111111111 zeta[49,1,2] + 0.1111111111111111 zeta[49,2,2]
   + 0.1111111111111111 zeta[49,3,2] + 0.1111111111111111 zeta[49,4,2]
   + 0.1111111111111111 zeta[49,5,2] + 0.1111111111111111 zeta[49,6,2]
   + 0.1111111111111111 zeta[49,7,2] - 0.25 zeta[45,1,2]
   - 0.25 zeta[45,2,2] - 0.25 zeta[45,3,2] - 0.25 zeta[45,4,2]
   - 0.25 zeta[45,5,2] - 0.25 zeta[45,6,2] - 0.25 zeta[45,7,2]
   - 0.25 zeta[23,1,2] - 0.25 zeta[23,2,2] - 0.25 zeta[23,3,2]
   - 0.25 zeta[23,4,2] - 0.25 zeta[23,5,2] - 0.25 zeta[23,6,2]
   - 0.25 zeta[23,7,2] + 0.1111111111111111 zeta[25,1,2]
   + 0.1111111111111111 zeta[25,2,2] + 0.1111111111111111 zeta[25,3,2]
   + 0.1111111111111111 zeta[25,4,2] + 0.1111111111111111 zeta[25,5,2]
   + 0.1111111111111111 zeta[25,6,2] + 0.1111111111111111 zeta[25,7,2]
   + 0.1111111111111111 zeta[26,1,2] + 0.1111111111111111 zeta[26,2,2]
   + 0.1111111111111111 zeta[26,3,2] + 0.1111111111111111 zeta[26,4,2]
   + 0.1111111111111111 zeta[26,5,2] + 0.1111111111111111 zeta[26,6,2]
   + 0.1111111111111111 zeta[26,7,2] + 0.1111111111111111 zeta[41,1,2]
   + 0.1111111111111111 zeta[41,2,2] + 0.1111111111111111 zeta[41,3,2]
   + 0.1111111111111111 zeta[41,4,2] + 0.1111111111111111 zeta[41,5,2]
   + 0.1111111111111111 zeta[41,6,2] + 0.1111111111111111 zeta[41,7,2]
   - 0.25 zeta[0,1,2] - 0.25 zeta[0,2,2] - 0.25 zeta[0,3,2]
   - 0.25 zeta[0,4,2] - 0.25 zeta[0,5,2] - 0.25 zeta[0,6,2]
   - 0.25 zeta[0,7,2] <= 0.01
 R665: 0.1111111111111111 zeta[43,1,2] + 0.1111111111111111 zeta[43,2,2]
   + 0.1111111111111111 zeta[43,3,2] + 0.1111111111111111 zeta[43,4,2]
   + 0.1111111111111111 zeta[43,5,2] + 0.1111111111111111 zeta[43,6,2]
   + 0.1111111111111111 zeta[43,7,2] + 0.1111111111111111 zeta[1,1,2]
   + 0.1111111111111111 zeta[1,2,2] + 0.1111111111111111 zeta[1,3,2]
   + 0.1111111111111111 zeta[1,4,2] + 0.1111111111111111 zeta[1,5,2]
   + 0.1111111111111111 zeta[1,6,2] + 0.1111111111111111 zeta[1,7,2]
   + 0.1111111111111111 zeta[46,1,2] + 0.1111111111111111 zeta[46,2,2]
   + 0.1111111111111111 zeta[46,3,2] + 0.1111111111111111 zeta[46,4,2]
   + 0.1111111111111111 zeta[46,5,2] + 0.1111111111111111 zeta[46,6,2]
   + 0.1111111111111111 zeta[46,7,2] + 0.1111111111111111 zeta[38,1,2]
   + 0.1111111111111111 zeta[38,2,2] + 0.1111111111111111 zeta[38,3,2]
   + 0.1111111111111111 zeta[38,4,2] + 0.1111111111111111 zeta[38,5,2]
   + 0.1111111111111111 zeta[38,6,2] + 0.1111111111111111 zeta[38,7,2]
   - 0.25 zeta[3,1,2] - 0.25 zeta[3,2,2] - 0.25 zeta[3,3,2]
   - 0.25 zeta[3,4,2] - 0.25 zeta[3,5,2] - 0.25 zeta[3,6,2]
   - 0.25 zeta[3,7,2] + 0.1111111111111111 zeta[24,1,2]
   + 0.1111111111111111 zeta[24,2,2] + 0.1111111111111111 zeta[24,3,2]
   + 0.1111111111111111 zeta[24,4,2] + 0.1111111111111111 zeta[24,5,2]
   + 0.1111111111111111 zeta[24,6,2] + 0.1111111111111111 zeta[24,7,2]
   + 0.1111111111111111 zeta[49,1,2] + 0.1111111111111111 zeta[49,2,2]
   + 0.1111111111111111 zeta[49,3,2] + 0.1111111111111111 zeta[49,4,2]
   + 0.1111111111111111 zeta[49,5,2] + 0.1111111111111111 zeta[49,6,2]
   + 0.1111111111111111 zeta[49,7,2] - 0.25 zeta[45,1,2]
   - 0.25 zeta[45,2,2] - 0.25 zeta[45,3,2] - 0.25 zeta[45,4,2]
   - 0.25 zeta[45,5,2] - 0.25 zeta[45,6,2] - 0.25 zeta[45,7,2]
   - 0.25 zeta[23,1,2] - 0.25 zeta[23,2,2] - 0.25 zeta[23,3,2]
   - 0.25 zeta[23,4,2] - 0.25 zeta[23,5,2] - 0.25 zeta[23,6,2]
   - 0.25 zeta[23,7,2] + 0.1111111111111111 zeta[25,1,2]
   + 0.1111111111111111 zeta[25,2,2] + 0.1111111111111111 zeta[25,3,2]
   + 0.1111111111111111 zeta[25,4,2] + 0.1111111111111111 zeta[25,5,2]
   + 0.1111111111111111 zeta[25,6,2] + 0.1111111111111111 zeta[25,7,2]
   + 0.1111111111111111 zeta[26,1,2] + 0.1111111111111111 zeta[26,2,2]
   + 0.1111111111111111 zeta[26,3,2] + 0.1111111111111111 zeta[26,4,2]
   + 0.1111111111111111 zeta[26,5,2] + 0.1111111111111111 zeta[26,6,2]
   + 0.1111111111111111 zeta[26,7,2] + 0.1111111111111111 zeta[41,1,2]
   + 0.1111111111111111 zeta[41,2,2] + 0.1111111111111111 zeta[41,3,2]
   + 0.1111111111111111 zeta[41,4,2] + 0.1111111111111111 zeta[41,5,2]
   + 0.1111111111111111 zeta[41,6,2] + 0.1111111111111111 zeta[41,7,2]
   - 0.25 zeta[0,1,2] - 0.25 zeta[0,2,2] - 0.25 zeta[0,3,2]
   - 0.25 zeta[0,4,2] - 0.25 zeta[0,5,2] - 0.25 zeta[0,6,2]
   - 0.25 zeta[0,7,2] >= -0.01
 R666: 0.1111111111111111 zeta[43,1,2] + 0.1111111111111111 zeta[43,2,2]
   + 0.1111111111111111 zeta[43,3,2] + 0.1111111111111111 zeta[43,4,2]
   + 0.1111111111111111 zeta[43,5,2] + 0.1111111111111111 zeta[43,6,2]
   + 0.1111111111111111 zeta[43,7,2] + 0.1111111111111111 zeta[1,1,2]
   + 0.1111111111111111 zeta[1,2,2] + 0.1111111111111111 zeta[1,3,2]
   + 0.1111111111111111 zeta[1,4,2] + 0.1111111111111111 zeta[1,5,2]
   + 0.1111111111111111 zeta[1,6,2] + 0.1111111111111111 zeta[1,7,2]
   + 0.1111111111111111 zeta[46,1,2] + 0.1111111111111111 zeta[46,2,2]
   + 0.1111111111111111 zeta[46,3,2] + 0.1111111111111111 zeta[46,4,2]
   + 0.1111111111111111 zeta[46,5,2] + 0.1111111111111111 zeta[46,6,2]
   + 0.1111111111111111 zeta[46,7,2] + 0.1111111111111111 zeta[38,1,2]
   + 0.1111111111111111 zeta[38,2,2] + 0.1111111111111111 zeta[38,3,2]
   + 0.1111111111111111 zeta[38,4,2] + 0.1111111111111111 zeta[38,5,2]
   + 0.1111111111111111 zeta[38,6,2] + 0.1111111111111111 zeta[38,7,2]
   + 0.1111111111111111 zeta[24,1,2] + 0.1111111111111111 zeta[24,2,2]
   + 0.1111111111111111 zeta[24,3,2] + 0.1111111111111111 zeta[24,4,2]
   + 0.1111111111111111 zeta[24,5,2] + 0.1111111111111111 zeta[24,6,2]
   + 0.1111111111111111 zeta[24,7,2] - 0.5 zeta[21,1,2] - 0.5 zeta[21,2,2]
   - 0.5 zeta[21,3,2] - 0.5 zeta[21,4,2] - 0.5 zeta[21,5,2]
   - 0.5 zeta[21,6,2] - 0.5 zeta[21,7,2] + 0.1111111111111111 zeta[49,1,2]
   + 0.1111111111111111 zeta[49,2,2] + 0.1111111111111111 zeta[49,3,2]
   + 0.1111111111111111 zeta[49,4,2] + 0.1111111111111111 zeta[49,5,2]
   + 0.1111111111111111 zeta[49,6,2] + 0.1111111111111111 zeta[49,7,2]
   - 0.5 zeta[48,1,2] - 0.5 zeta[48,2,2] - 0.5 zeta[48,3,2]
   - 0.5 zeta[48,4,2] - 0.5 zeta[48,5,2] - 0.5 zeta[48,6,2]
   - 0.5 zeta[48,7,2] + 0.1111111111111111 zeta[25,1,2]
   + 0.1111111111111111 zeta[25,2,2] + 0.1111111111111111 zeta[25,3,2]
   + 0.1111111111111111 zeta[25,4,2] + 0.1111111111111111 zeta[25,5,2]
   + 0.1111111111111111 zeta[25,6,2] + 0.1111111111111111 zeta[25,7,2]
   + 0.1111111111111111 zeta[26,1,2] + 0.1111111111111111 zeta[26,2,2]
   + 0.1111111111111111 zeta[26,3,2] + 0.1111111111111111 zeta[26,4,2]
   + 0.1111111111111111 zeta[26,5,2] + 0.1111111111111111 zeta[26,6,2]
   + 0.1111111111111111 zeta[26,7,2] + 0.1111111111111111 zeta[41,1,2]
   + 0.1111111111111111 zeta[41,2,2] + 0.1111111111111111 zeta[41,3,2]
   + 0.1111111111111111 zeta[41,4,2] + 0.1111111111111111 zeta[41,5,2]
   + 0.1111111111111111 zeta[41,6,2] + 0.1111111111111111 zeta[41,7,2]
   <= 0.01
 R667: 0.1111111111111111 zeta[43,1,2] + 0.1111111111111111 zeta[43,2,2]
   + 0.1111111111111111 zeta[43,3,2] + 0.1111111111111111 zeta[43,4,2]
   + 0.1111111111111111 zeta[43,5,2] + 0.1111111111111111 zeta[43,6,2]
   + 0.1111111111111111 zeta[43,7,2] + 0.1111111111111111 zeta[1,1,2]
   + 0.1111111111111111 zeta[1,2,2] + 0.1111111111111111 zeta[1,3,2]
   + 0.1111111111111111 zeta[1,4,2] + 0.1111111111111111 zeta[1,5,2]
   + 0.1111111111111111 zeta[1,6,2] + 0.1111111111111111 zeta[1,7,2]
   + 0.1111111111111111 zeta[46,1,2] + 0.1111111111111111 zeta[46,2,2]
   + 0.1111111111111111 zeta[46,3,2] + 0.1111111111111111 zeta[46,4,2]
   + 0.1111111111111111 zeta[46,5,2] + 0.1111111111111111 zeta[46,6,2]
   + 0.1111111111111111 zeta[46,7,2] + 0.1111111111111111 zeta[38,1,2]
   + 0.1111111111111111 zeta[38,2,2] + 0.1111111111111111 zeta[38,3,2]
   + 0.1111111111111111 zeta[38,4,2] + 0.1111111111111111 zeta[38,5,2]
   + 0.1111111111111111 zeta[38,6,2] + 0.1111111111111111 zeta[38,7,2]
   + 0.1111111111111111 zeta[24,1,2] + 0.1111111111111111 zeta[24,2,2]
   + 0.1111111111111111 zeta[24,3,2] + 0.1111111111111111 zeta[24,4,2]
   + 0.1111111111111111 zeta[24,5,2] + 0.1111111111111111 zeta[24,6,2]
   + 0.1111111111111111 zeta[24,7,2] - 0.5 zeta[21,1,2] - 0.5 zeta[21,2,2]
   - 0.5 zeta[21,3,2] - 0.5 zeta[21,4,2] - 0.5 zeta[21,5,2]
   - 0.5 zeta[21,6,2] - 0.5 zeta[21,7,2] + 0.1111111111111111 zeta[49,1,2]
   + 0.1111111111111111 zeta[49,2,2] + 0.1111111111111111 zeta[49,3,2]
   + 0.1111111111111111 zeta[49,4,2] + 0.1111111111111111 zeta[49,5,2]
   + 0.1111111111111111 zeta[49,6,2] + 0.1111111111111111 zeta[49,7,2]
   - 0.5 zeta[48,1,2] - 0.5 zeta[48,2,2] - 0.5 zeta[48,3,2]
   - 0.5 zeta[48,4,2] - 0.5 zeta[48,5,2] - 0.5 zeta[48,6,2]
   - 0.5 zeta[48,7,2] + 0.1111111111111111 zeta[25,1,2]
   + 0.1111111111111111 zeta[25,2,2] + 0.1111111111111111 zeta[25,3,2]
   + 0.1111111111111111 zeta[25,4,2] + 0.1111111111111111 zeta[25,5,2]
   + 0.1111111111111111 zeta[25,6,2] + 0.1111111111111111 zeta[25,7,2]
   + 0.1111111111111111 zeta[26,1,2] + 0.1111111111111111 zeta[26,2,2]
   + 0.1111111111111111 zeta[26,3,2] + 0.1111111111111111 zeta[26,4,2]
   + 0.1111111111111111 zeta[26,5,2] + 0.1111111111111111 zeta[26,6,2]
   + 0.1111111111111111 zeta[26,7,2] + 0.1111111111111111 zeta[41,1,2]
   + 0.1111111111111111 zeta[41,2,2] + 0.1111111111111111 zeta[41,3,2]
   + 0.1111111111111111 zeta[41,4,2] + 0.1111111111111111 zeta[41,5,2]
   + 0.1111111111111111 zeta[41,6,2] + 0.1111111111111111 zeta[41,7,2]
   >= -0.01
 R668: 0.1111111111111111 zeta[16,1,2] + 0.1111111111111111 zeta[16,2,2]
   + 0.1111111111111111 zeta[16,3,2] + 0.1111111111111111 zeta[16,4,2]
   + 0.1111111111111111 zeta[16,5,2] + 0.1111111111111111 zeta[16,6,2]
   + 0.1111111111111111 zeta[16,7,2] + 0.1111111111111111 zeta[27,1,2]
   + 0.1111111111111111 zeta[27,2,2] + 0.1111111111111111 zeta[27,3,2]
   + 0.1111111111111111 zeta[27,4,2] + 0.1111111111111111 zeta[27,5,2]
   + 0.1111111111111111 zeta[27,6,2] + 0.1111111111111111 zeta[27,7,2]
   - 0.25 zeta[3,1,2] - 0.25 zeta[3,2,2] - 0.25 zeta[3,3,2]
   - 0.25 zeta[3,4,2] - 0.25 zeta[3,5,2] - 0.25 zeta[3,6,2]
   - 0.25 zeta[3,7,2] + 0.1111111111111111 zeta[34,1,2]
   + 0.1111111111111111 zeta[34,2,2] + 0.1111111111111111 zeta[34,3,2]
   + 0.1111111111111111 zeta[34,4,2] + 0.1111111111111111 zeta[34,5,2]
   + 0.1111111111111111 zeta[34,6,2] + 0.1111111111111111 zeta[34,7,2]
   + 0.1111111111111111 zeta[9,1,2] + 0.1111111111111111 zeta[9,2,2]
   + 0.1111111111111111 zeta[9,3,2] + 0.1111111111111111 zeta[9,4,2]
   + 0.1111111111111111 zeta[9,5,2] + 0.1111111111111111 zeta[9,6,2]
   + 0.1111111111111111 zeta[9,7,2] - 0.25 zeta[45,1,2] - 0.25 zeta[45,2,2]
   - 0.25 zeta[45,3,2] - 0.25 zeta[45,4,2] - 0.25 zeta[45,5,2]
   - 0.25 zeta[45,6,2] - 0.25 zeta[45,7,2] - 0.25 zeta[23,1,2]
   - 0.25 zeta[23,2,2] - 0.25 zeta[23,3,2] - 0.25 zeta[23,4,2]
   - 0.25 zeta[23,5,2] - 0.25 zeta[23,6,2] - 0.25 zeta[23,7,2]
   + 0.1111111111111111 zeta[19,1,2] + 0.1111111111111111 zeta[19,2,2]
   + 0.1111111111111111 zeta[19,3,2] + 0.1111111111111111 zeta[19,4,2]
   + 0.1111111111111111 zeta[19,5,2] + 0.1111111111111111 zeta[19,6,2]
   + 0.1111111111111111 zeta[19,7,2] + 0.1111111111111111 zeta[6,1,2]
   + 0.1111111111111111 zeta[6,2,2] + 0.1111111111111111 zeta[6,3,2]
   + 0.1111111111111111 zeta[6,4,2] + 0.1111111111111111 zeta[6,5,2]
   + 0.1111111111111111 zeta[6,6,2] + 0.1111111111111111 zeta[6,7,2]
   + 0.1111111111111111 zeta[4,1,2] + 0.1111111111111111 zeta[4,2,2]
   + 0.1111111111111111 zeta[4,3,2] + 0.1111111111111111 zeta[4,4,2]
   + 0.1111111111111111 zeta[4,5,2] + 0.1111111111111111 zeta[4,6,2]
   + 0.1111111111111111 zeta[4,7,2] + 0.1111111111111111 zeta[31,1,2]
   + 0.1111111111111111 zeta[31,2,2] + 0.1111111111111111 zeta[31,3,2]
   + 0.1111111111111111 zeta[31,4,2] + 0.1111111111111111 zeta[31,5,2]
   + 0.1111111111111111 zeta[31,6,2] + 0.1111111111111111 zeta[31,7,2]
   + 0.1111111111111111 zeta[7,1,2] + 0.1111111111111111 zeta[7,2,2]
   + 0.1111111111111111 zeta[7,3,2] + 0.1111111111111111 zeta[7,4,2]
   + 0.1111111111111111 zeta[7,5,2] + 0.1111111111111111 zeta[7,6,2]
   + 0.1111111111111111 zeta[7,7,2] - 0.25 zeta[0,1,2] - 0.25 zeta[0,2,2]
   - 0.25 zeta[0,3,2] - 0.25 zeta[0,4,2] - 0.25 zeta[0,5,2]
   - 0.25 zeta[0,6,2] - 0.25 zeta[0,7,2] <= 0.01
 R669: 0.1111111111111111 zeta[16,1,2] + 0.1111111111111111 zeta[16,2,2]
   + 0.1111111111111111 zeta[16,3,2] + 0.1111111111111111 zeta[16,4,2]
   + 0.1111111111111111 zeta[16,5,2] + 0.1111111111111111 zeta[16,6,2]
   + 0.1111111111111111 zeta[16,7,2] + 0.1111111111111111 zeta[27,1,2]
   + 0.1111111111111111 zeta[27,2,2] + 0.1111111111111111 zeta[27,3,2]
   + 0.1111111111111111 zeta[27,4,2] + 0.1111111111111111 zeta[27,5,2]
   + 0.1111111111111111 zeta[27,6,2] + 0.1111111111111111 zeta[27,7,2]
   - 0.25 zeta[3,1,2] - 0.25 zeta[3,2,2] - 0.25 zeta[3,3,2]
   - 0.25 zeta[3,4,2] - 0.25 zeta[3,5,2] - 0.25 zeta[3,6,2]
   - 0.25 zeta[3,7,2] + 0.1111111111111111 zeta[34,1,2]
   + 0.1111111111111111 zeta[34,2,2] + 0.1111111111111111 zeta[34,3,2]
   + 0.1111111111111111 zeta[34,4,2] + 0.1111111111111111 zeta[34,5,2]
   + 0.1111111111111111 zeta[34,6,2] + 0.1111111111111111 zeta[34,7,2]
   + 0.1111111111111111 zeta[9,1,2] + 0.1111111111111111 zeta[9,2,2]
   + 0.1111111111111111 zeta[9,3,2] + 0.1111111111111111 zeta[9,4,2]
   + 0.1111111111111111 zeta[9,5,2] + 0.1111111111111111 zeta[9,6,2]
   + 0.1111111111111111 zeta[9,7,2] - 0.25 zeta[45,1,2] - 0.25 zeta[45,2,2]
   - 0.25 zeta[45,3,2] - 0.25 zeta[45,4,2] - 0.25 zeta[45,5,2]
   - 0.25 zeta[45,6,2] - 0.25 zeta[45,7,2] - 0.25 zeta[23,1,2]
   - 0.25 zeta[23,2,2] - 0.25 zeta[23,3,2] - 0.25 zeta[23,4,2]
   - 0.25 zeta[23,5,2] - 0.25 zeta[23,6,2] - 0.25 zeta[23,7,2]
   + 0.1111111111111111 zeta[19,1,2] + 0.1111111111111111 zeta[19,2,2]
   + 0.1111111111111111 zeta[19,3,2] + 0.1111111111111111 zeta[19,4,2]
   + 0.1111111111111111 zeta[19,5,2] + 0.1111111111111111 zeta[19,6,2]
   + 0.1111111111111111 zeta[19,7,2] + 0.1111111111111111 zeta[6,1,2]
   + 0.1111111111111111 zeta[6,2,2] + 0.1111111111111111 zeta[6,3,2]
   + 0.1111111111111111 zeta[6,4,2] + 0.1111111111111111 zeta[6,5,2]
   + 0.1111111111111111 zeta[6,6,2] + 0.1111111111111111 zeta[6,7,2]
   + 0.1111111111111111 zeta[4,1,2] + 0.1111111111111111 zeta[4,2,2]
   + 0.1111111111111111 zeta[4,3,2] + 0.1111111111111111 zeta[4,4,2]
   + 0.1111111111111111 zeta[4,5,2] + 0.1111111111111111 zeta[4,6,2]
   + 0.1111111111111111 zeta[4,7,2] + 0.1111111111111111 zeta[31,1,2]
   + 0.1111111111111111 zeta[31,2,2] + 0.1111111111111111 zeta[31,3,2]
   + 0.1111111111111111 zeta[31,4,2] + 0.1111111111111111 zeta[31,5,2]
   + 0.1111111111111111 zeta[31,6,2] + 0.1111111111111111 zeta[31,7,2]
   + 0.1111111111111111 zeta[7,1,2] + 0.1111111111111111 zeta[7,2,2]
   + 0.1111111111111111 zeta[7,3,2] + 0.1111111111111111 zeta[7,4,2]
   + 0.1111111111111111 zeta[7,5,2] + 0.1111111111111111 zeta[7,6,2]
   + 0.1111111111111111 zeta[7,7,2] - 0.25 zeta[0,1,2] - 0.25 zeta[0,2,2]
   - 0.25 zeta[0,3,2] - 0.25 zeta[0,4,2] - 0.25 zeta[0,5,2]
   - 0.25 zeta[0,6,2] - 0.25 zeta[0,7,2] >= -0.01
 R670: 0.1111111111111111 zeta[16,1,2] + 0.1111111111111111 zeta[16,2,2]
   + 0.1111111111111111 zeta[16,3,2] + 0.1111111111111111 zeta[16,4,2]
   + 0.1111111111111111 zeta[16,5,2] + 0.1111111111111111 zeta[16,6,2]
   + 0.1111111111111111 zeta[16,7,2] + 0.1111111111111111 zeta[27,1,2]
   + 0.1111111111111111 zeta[27,2,2] + 0.1111111111111111 zeta[27,3,2]
   + 0.1111111111111111 zeta[27,4,2] + 0.1111111111111111 zeta[27,5,2]
   + 0.1111111111111111 zeta[27,6,2] + 0.1111111111111111 zeta[27,7,2]
   - 0.5 zeta[21,1,2] - 0.5 zeta[21,2,2] - 0.5 zeta[21,3,2]
   - 0.5 zeta[21,4,2] - 0.5 zeta[21,5,2] - 0.5 zeta[21,6,2]
   - 0.5 zeta[21,7,2] - 0.5 zeta[48,1,2] - 0.5 zeta[48,2,2]
   - 0.5 zeta[48,3,2] - 0.5 zeta[48,4,2] - 0.5 zeta[48,5,2]
   - 0.5 zeta[48,6,2] - 0.5 zeta[48,7,2] + 0.1111111111111111 zeta[34,1,2]
   + 0.1111111111111111 zeta[34,2,2] + 0.1111111111111111 zeta[34,3,2]
   + 0.1111111111111111 zeta[34,4,2] + 0.1111111111111111 zeta[34,5,2]
   + 0.1111111111111111 zeta[34,6,2] + 0.1111111111111111 zeta[34,7,2]
   + 0.1111111111111111 zeta[9,1,2] + 0.1111111111111111 zeta[9,2,2]
   + 0.1111111111111111 zeta[9,3,2] + 0.1111111111111111 zeta[9,4,2]
   + 0.1111111111111111 zeta[9,5,2] + 0.1111111111111111 zeta[9,6,2]
   + 0.1111111111111111 zeta[9,7,2] + 0.1111111111111111 zeta[19,1,2]
   + 0.1111111111111111 zeta[19,2,2] + 0.1111111111111111 zeta[19,3,2]
   + 0.1111111111111111 zeta[19,4,2] + 0.1111111111111111 zeta[19,5,2]
   + 0.1111111111111111 zeta[19,6,2] + 0.1111111111111111 zeta[19,7,2]
   + 0.1111111111111111 zeta[6,1,2] + 0.1111111111111111 zeta[6,2,2]
   + 0.1111111111111111 zeta[6,3,2] + 0.1111111111111111 zeta[6,4,2]
   + 0.1111111111111111 zeta[6,5,2] + 0.1111111111111111 zeta[6,6,2]
   + 0.1111111111111111 zeta[6,7,2] + 0.1111111111111111 zeta[4,1,2]
   + 0.1111111111111111 zeta[4,2,2] + 0.1111111111111111 zeta[4,3,2]
   + 0.1111111111111111 zeta[4,4,2] + 0.1111111111111111 zeta[4,5,2]
   + 0.1111111111111111 zeta[4,6,2] + 0.1111111111111111 zeta[4,7,2]
   + 0.1111111111111111 zeta[31,1,2] + 0.1111111111111111 zeta[31,2,2]
   + 0.1111111111111111 zeta[31,3,2] + 0.1111111111111111 zeta[31,4,2]
   + 0.1111111111111111 zeta[31,5,2] + 0.1111111111111111 zeta[31,6,2]
   + 0.1111111111111111 zeta[31,7,2] + 0.1111111111111111 zeta[7,1,2]
   + 0.1111111111111111 zeta[7,2,2] + 0.1111111111111111 zeta[7,3,2]
   + 0.1111111111111111 zeta[7,4,2] + 0.1111111111111111 zeta[7,5,2]
   + 0.1111111111111111 zeta[7,6,2] + 0.1111111111111111 zeta[7,7,2]
   <= 0.01
 R671: 0.1111111111111111 zeta[16,1,2] + 0.1111111111111111 zeta[16,2,2]
   + 0.1111111111111111 zeta[16,3,2] + 0.1111111111111111 zeta[16,4,2]
   + 0.1111111111111111 zeta[16,5,2] + 0.1111111111111111 zeta[16,6,2]
   + 0.1111111111111111 zeta[16,7,2] + 0.1111111111111111 zeta[27,1,2]
   + 0.1111111111111111 zeta[27,2,2] + 0.1111111111111111 zeta[27,3,2]
   + 0.1111111111111111 zeta[27,4,2] + 0.1111111111111111 zeta[27,5,2]
   + 0.1111111111111111 zeta[27,6,2] + 0.1111111111111111 zeta[27,7,2]
   - 0.5 zeta[21,1,2] - 0.5 zeta[21,2,2] - 0.5 zeta[21,3,2]
   - 0.5 zeta[21,4,2] - 0.5 zeta[21,5,2] - 0.5 zeta[21,6,2]
   - 0.5 zeta[21,7,2] - 0.5 zeta[48,1,2] - 0.5 zeta[48,2,2]
   - 0.5 zeta[48,3,2] - 0.5 zeta[48,4,2] - 0.5 zeta[48,5,2]
   - 0.5 zeta[48,6,2] - 0.5 zeta[48,7,2] + 0.1111111111111111 zeta[34,1,2]
   + 0.1111111111111111 zeta[34,2,2] + 0.1111111111111111 zeta[34,3,2]
   + 0.1111111111111111 zeta[34,4,2] + 0.1111111111111111 zeta[34,5,2]
   + 0.1111111111111111 zeta[34,6,2] + 0.1111111111111111 zeta[34,7,2]
   + 0.1111111111111111 zeta[9,1,2] + 0.1111111111111111 zeta[9,2,2]
   + 0.1111111111111111 zeta[9,3,2] + 0.1111111111111111 zeta[9,4,2]
   + 0.1111111111111111 zeta[9,5,2] + 0.1111111111111111 zeta[9,6,2]
   + 0.1111111111111111 zeta[9,7,2] + 0.1111111111111111 zeta[19,1,2]
   + 0.1111111111111111 zeta[19,2,2] + 0.1111111111111111 zeta[19,3,2]
   + 0.1111111111111111 zeta[19,4,2] + 0.1111111111111111 zeta[19,5,2]
   + 0.1111111111111111 zeta[19,6,2] + 0.1111111111111111 zeta[19,7,2]
   + 0.1111111111111111 zeta[6,1,2] + 0.1111111111111111 zeta[6,2,2]
   + 0.1111111111111111 zeta[6,3,2] + 0.1111111111111111 zeta[6,4,2]
   + 0.1111111111111111 zeta[6,5,2] + 0.1111111111111111 zeta[6,6,2]
   + 0.1111111111111111 zeta[6,7,2] + 0.1111111111111111 zeta[4,1,2]
   + 0.1111111111111111 zeta[4,2,2] + 0.1111111111111111 zeta[4,3,2]
   + 0.1111111111111111 zeta[4,4,2] + 0.1111111111111111 zeta[4,5,2]
   + 0.1111111111111111 zeta[4,6,2] + 0.1111111111111111 zeta[4,7,2]
   + 0.1111111111111111 zeta[31,1,2] + 0.1111111111111111 zeta[31,2,2]
   + 0.1111111111111111 zeta[31,3,2] + 0.1111111111111111 zeta[31,4,2]
   + 0.1111111111111111 zeta[31,5,2] + 0.1111111111111111 zeta[31,6,2]
   + 0.1111111111111111 zeta[31,7,2] + 0.1111111111111111 zeta[7,1,2]
   + 0.1111111111111111 zeta[7,2,2] + 0.1111111111111111 zeta[7,3,2]
   + 0.1111111111111111 zeta[7,4,2] + 0.1111111111111111 zeta[7,5,2]
   + 0.1111111111111111 zeta[7,6,2] + 0.1111111111111111 zeta[7,7,2]
   >= -0.01
 R672: 0.25 zeta[3,1,2] + 0.25 zeta[3,2,2] + 0.25 zeta[3,3,2]
   + 0.25 zeta[3,4,2] + 0.25 zeta[3,5,2] + 0.25 zeta[3,6,2]
   + 0.25 zeta[3,7,2] - 0.5 zeta[21,1,2] - 0.5 zeta[21,2,2]
   - 0.5 zeta[21,3,2] - 0.5 zeta[21,4,2] - 0.5 zeta[21,5,2]
   - 0.5 zeta[21,6,2] - 0.5 zeta[21,7,2] - 0.5 zeta[48,1,2]
   - 0.5 zeta[48,2,2] - 0.5 zeta[48,3,2] - 0.5 zeta[48,4,2]
   - 0.5 zeta[48,5,2] - 0.5 zeta[48,6,2] - 0.5 zeta[48,7,2]
   + 0.25 zeta[45,1,2] + 0.25 zeta[45,2,2] + 0.25 zeta[45,3,2]
   + 0.25 zeta[45,4,2] + 0.25 zeta[45,5,2] + 0.25 zeta[45,6,2]
   + 0.25 zeta[45,7,2] + 0.25 zeta[23,1,2] + 0.25 zeta[23,2,2]
   + 0.25 zeta[23,3,2] + 0.25 zeta[23,4,2] + 0.25 zeta[23,5,2]
   + 0.25 zeta[23,6,2] + 0.25 zeta[23,7,2] + 0.25 zeta[0,1,2]
   + 0.25 zeta[0,2,2] + 0.25 zeta[0,3,2] + 0.25 zeta[0,4,2]
   + 0.25 zeta[0,5,2] + 0.25 zeta[0,6,2] + 0.25 zeta[0,7,2] <= 0.01
 R673: 0.25 zeta[3,1,2] + 0.25 zeta[3,2,2] + 0.25 zeta[3,3,2]
   + 0.25 zeta[3,4,2] + 0.25 zeta[3,5,2] + 0.25 zeta[3,6,2]
   + 0.25 zeta[3,7,2] - 0.5 zeta[21,1,2] - 0.5 zeta[21,2,2]
   - 0.5 zeta[21,3,2] - 0.5 zeta[21,4,2] - 0.5 zeta[21,5,2]
   - 0.5 zeta[21,6,2] - 0.5 zeta[21,7,2] - 0.5 zeta[48,1,2]
   - 0.5 zeta[48,2,2] - 0.5 zeta[48,3,2] - 0.5 zeta[48,4,2]
   - 0.5 zeta[48,5,2] - 0.5 zeta[48,6,2] - 0.5 zeta[48,7,2]
   + 0.25 zeta[45,1,2] + 0.25 zeta[45,2,2] + 0.25 zeta[45,3,2]
   + 0.25 zeta[45,4,2] + 0.25 zeta[45,5,2] + 0.25 zeta[45,6,2]
   + 0.25 zeta[45,7,2] + 0.25 zeta[23,1,2] + 0.25 zeta[23,2,2]
   + 0.25 zeta[23,3,2] + 0.25 zeta[23,4,2] + 0.25 zeta[23,5,2]
   + 0.25 zeta[23,6,2] + 0.25 zeta[23,7,2] + 0.25 zeta[0,1,2]
   + 0.25 zeta[0,2,2] + 0.25 zeta[0,3,2] + 0.25 zeta[0,4,2]
   + 0.25 zeta[0,5,2] + 0.25 zeta[0,6,2] + 0.25 zeta[0,7,2] >= -0.01
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
 p[4] p[5] p[6] p[7] zeta[43,1,1] zeta[43,1,2] zeta[43,2,1] zeta[43,2,2]
 zeta[43,3,1] zeta[43,3,2] zeta[43,4,1] zeta[43,4,2] zeta[43,5,1]
 zeta[43,5,2] zeta[43,6,1] zeta[43,6,2] zeta[43,7,1] zeta[43,7,2]
 zeta[1,1,1] zeta[1,1,2] zeta[1,2,1] zeta[1,2,2] zeta[1,3,1] zeta[1,3,2]
 zeta[1,4,1] zeta[1,4,2] zeta[1,5,1] zeta[1,5,2] zeta[1,6,1] zeta[1,6,2]
 zeta[1,7,1] zeta[1,7,2] zeta[46,1,1] zeta[46,1,2] zeta[46,2,1]
 zeta[46,2,2] zeta[46,3,1] zeta[46,3,2] zeta[46,4,1] zeta[46,4,2]
 zeta[46,5,1] zeta[46,5,2] zeta[46,6,1] zeta[46,6,2] zeta[46,7,1]
 zeta[46,7,2] zeta[38,1,1] zeta[38,1,2] zeta[38,2,1] zeta[38,2,2]
 zeta[38,3,1] zeta[38,3,2] zeta[38,4,1] zeta[38,4,2] zeta[38,5,1]
 zeta[38,5,2] zeta[38,6,1] zeta[38,6,2] zeta[38,7,1] zeta[38,7,2]
 zeta[16,1,1] zeta[16,1,2] zeta[16,2,1] zeta[16,2,2] zeta[16,3,1]
 zeta[16,3,2] zeta[16,4,1] zeta[16,4,2] zeta[16,5,1] zeta[16,5,2]
 zeta[16,6,1] zeta[16,6,2] zeta[16,7,1] zeta[16,7,2] zeta[27,1,1]
 zeta[27,1,2] zeta[27,2,1] zeta[27,2,2] zeta[27,3,1] zeta[27,3,2]
 zeta[27,4,1] zeta[27,4,2] zeta[27,5,1] zeta[27,5,2] zeta[27,6,1]
 zeta[27,6,2] zeta[27,7,1] zeta[27,7,2] zeta[3,1,1] zeta[3,1,2] zeta[3,2,1]
 zeta[3,2,2] zeta[3,3,1] zeta[3,3,2] zeta[3,4,1] zeta[3,4,2] zeta[3,5,1]
 zeta[3,5,2] zeta[3,6,1] zeta[3,6,2] zeta[3,7,1] zeta[3,7,2] zeta[24,1,1]
 zeta[24,1,2] zeta[24,2,1] zeta[24,2,2] zeta[24,3,1] zeta[24,3,2]
 zeta[24,4,1] zeta[24,4,2] zeta[24,5,1] zeta[24,5,2] zeta[24,6,1]
 zeta[24,6,2] zeta[24,7,1] zeta[24,7,2] zeta[21,1,1] zeta[21,1,2]
 zeta[21,2,1] zeta[21,2,2] zeta[21,3,1] zeta[21,3,2] zeta[21,4,1]
 zeta[21,4,2] zeta[21,5,1] zeta[21,5,2] zeta[21,6,1] zeta[21,6,2]
 zeta[21,7,1] zeta[21,7,2] zeta[49,1,1] zeta[49,1,2] zeta[49,2,1]
 zeta[49,2,2] zeta[49,3,1] zeta[49,3,2] zeta[49,4,1] zeta[49,4,2]
 zeta[49,5,1] zeta[49,5,2] zeta[49,6,1] zeta[49,6,2] zeta[49,7,1]
 zeta[49,7,2] zeta[48,1,1] zeta[48,1,2] zeta[48,2,1] zeta[48,2,2]
 zeta[48,3,1] zeta[48,3,2] zeta[48,4,1] zeta[48,4,2] zeta[48,5,1]
 zeta[48,5,2] zeta[48,6,1] zeta[48,6,2] zeta[48,7,1] zeta[48,7,2]
 zeta[34,1,1] zeta[34,1,2] zeta[34,2,1] zeta[34,2,2] zeta[34,3,1]
 zeta[34,3,2] zeta[34,4,1] zeta[34,4,2] zeta[34,5,1] zeta[34,5,2]
 zeta[34,6,1] zeta[34,6,2] zeta[34,7,1] zeta[34,7,2] zeta[9,1,1]
 zeta[9,1,2] zeta[9,2,1] zeta[9,2,2] zeta[9,3,1] zeta[9,3,2] zeta[9,4,1]
 zeta[9,4,2] zeta[9,5,1] zeta[9,5,2] zeta[9,6,1] zeta[9,6,2] zeta[9,7,1]
 zeta[9,7,2] zeta[45,1,1] zeta[45,1,2] zeta[45,2,1] zeta[45,2,2]
 zeta[45,3,1] zeta[45,3,2] zeta[45,4,1] zeta[45,4,2] zeta[45,5,1]
 zeta[45,5,2] zeta[45,6,1] zeta[45,6,2] zeta[45,7,1] zeta[45,7,2]
 zeta[23,1,1] zeta[23,1,2] zeta[23,2,1] zeta[23,2,2] zeta[23,3,1]
 zeta[23,3,2] zeta[23,4,1] zeta[23,4,2] zeta[23,5,1] zeta[23,5,2]
 zeta[23,6,1] zeta[23,6,2] zeta[23,7,1] zeta[23,7,2] zeta[19,1,1]
 zeta[19,1,2] zeta[19,2,1] zeta[19,2,2] zeta[19,3,1] zeta[19,3,2]
 zeta[19,4,1] zeta[19,4,2] zeta[19,5,1] zeta[19,5,2] zeta[19,6,1]
 zeta[19,6,2] zeta[19,7,1] zeta[19,7,2] zeta[6,1,1] zeta[6,1,2] zeta[6,2,1]
 zeta[6,2,2] zeta[6,3,1] zeta[6,3,2] zeta[6,4,1] zeta[6,4,2] zeta[6,5,1]
 zeta[6,5,2] zeta[6,6,1] zeta[6,6,2] zeta[6,7,1] zeta[6,7,2] zeta[25,1,1]
 zeta[25,1,2] zeta[25,2,1] zeta[25,2,2] zeta[25,3,1] zeta[25,3,2]
 zeta[25,4,1] zeta[25,4,2] zeta[25,5,1] zeta[25,5,2] zeta[25,6,1]
 zeta[25,6,2] zeta[25,7,1] zeta[25,7,2] zeta[4,1,1] zeta[4,1,2] zeta[4,2,1]
 zeta[4,2,2] zeta[4,3,1] zeta[4,3,2] zeta[4,4,1] zeta[4,4,2] zeta[4,5,1]
 zeta[4,5,2] zeta[4,6,1] zeta[4,6,2] zeta[4,7,1] zeta[4,7,2] zeta[26,1,1]
 zeta[26,1,2] zeta[26,2,1] zeta[26,2,2] zeta[26,3,1] zeta[26,3,2]
 zeta[26,4,1] zeta[26,4,2] zeta[26,5,1] zeta[26,5,2] zeta[26,6,1]
 zeta[26,6,2] zeta[26,7,1] zeta[26,7,2] zeta[41,1,1] zeta[41,1,2]
 zeta[41,2,1] zeta[41,2,2] zeta[41,3,1] zeta[41,3,2] zeta[41,4,1]
 zeta[41,4,2] zeta[41,5,1] zeta[41,5,2] zeta[41,6,1] zeta[41,6,2]
 zeta[41,7,1] zeta[41,7,2] zeta[31,1,1] zeta[31,1,2] zeta[31,2,1]
 zeta[31,2,2] zeta[31,3,1] zeta[31,3,2] zeta[31,4,1] zeta[31,4,2]
 zeta[31,5,1] zeta[31,5,2] zeta[31,6,1] zeta[31,6,2] zeta[31,7,1]
 zeta[31,7,2] zeta[7,1,1] zeta[7,1,2] zeta[7,2,1] zeta[7,2,2] zeta[7,3,1]
 zeta[7,3,2] zeta[7,4,1] zeta[7,4,2] zeta[7,5,1] zeta[7,5,2] zeta[7,6,1]
 zeta[7,6,2] zeta[7,7,1] zeta[7,7,2] zeta[0,1,1] zeta[0,1,2] zeta[0,2,1]
 zeta[0,2,2] zeta[0,3,1] zeta[0,3,2] zeta[0,4,1] zeta[0,4,2] zeta[0,5,1]
 zeta[0,5,2] zeta[0,6,1] zeta[0,6,2] zeta[0,7,1] zeta[0,7,2] z[43,1]
 z[43,2] z[43,3] z[43,4] z[43,5] z[43,6] z[43,7] z[1,1] z[1,2] z[1,3]
 z[1,4] z[1,5] z[1,6] z[1,7] z[46,1] z[46,2] z[46,3] z[46,4] z[46,5]
 z[46,6] z[46,7] z[38,1] z[38,2] z[38,3] z[38,4] z[38,5] z[38,6] z[38,7]
 z[16,1] z[16,2] z[16,3] z[16,4] z[16,5] z[16,6] z[16,7] z[27,1] z[27,2]
 z[27,3] z[27,4] z[27,5] z[27,6] z[27,7] z[3,1] z[3,2] z[3,3] z[3,4] z[3,5]
 z[3,6] z[3,7] z[24,1] z[24,2] z[24,3] z[24,4] z[24,5] z[24,6] z[24,7]
 z[21,1] z[21,2] z[21,3] z[21,4] z[21,5] z[21,6] z[21,7] z[49,1] z[49,2]
 z[49,3] z[49,4] z[49,5] z[49,6] z[49,7] z[48,1] z[48,2] z[48,3] z[48,4]
 z[48,5] z[48,6] z[48,7] z[34,1] z[34,2] z[34,3] z[34,4] z[34,5] z[34,6]
 z[34,7] z[9,1] z[9,2] z[9,3] z[9,4] z[9,5] z[9,6] z[9,7] z[45,1] z[45,2]
 z[45,3] z[45,4] z[45,5] z[45,6] z[45,7] z[23,1] z[23,2] z[23,3] z[23,4]
 z[23,5] z[23,6] z[23,7] z[19,1] z[19,2] z[19,3] z[19,4] z[19,5] z[19,6]
 z[19,7] z[6,1] z[6,2] z[6,3] z[6,4] z[6,5] z[6,6] z[6,7] z[25,1] z[25,2]
 z[25,3] z[25,4] z[25,5] z[25,6] z[25,7] z[4,1] z[4,2] z[4,3] z[4,4] z[4,5]
 z[4,6] z[4,7] z[26,1] z[26,2] z[26,3] z[26,4] z[26,5] z[26,6] z[26,7]
 z[41,1] z[41,2] z[41,3] z[41,4] z[41,5] z[41,6] z[41,7] z[31,1] z[31,2]
 z[31,3] z[31,4] z[31,5] z[31,6] z[31,7] z[7,1] z[7,2] z[7,3] z[7,4] z[7,5]
 z[7,6] z[7,7] z[0,1] z[0,2] z[0,3] z[0,4] z[0,5] z[0,6] z[0,7]
End
