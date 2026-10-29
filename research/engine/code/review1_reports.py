from review1_common import *
import pandas as pd

def main():
 check_review();out=ROOT/'results/review1';r=json.loads((out/'review_resolution.json').read_text());op=json.loads((out/'oracle_opportunity.json').read_text());w=pd.read_csv(out/'weight_and_deletion_sensitivity.csv');cost=pd.read_csv(out/'recovery_cost_inference.csv');refs=json.loads((ROOT/'audit/review1_reference_audit.json').read_text())
 response=f'''# Response to the supplied simulated review

This responds to the user-provided simulated critique, not an external editorial decision. Its numerical acceptance probabilities are subjective estimates and are not repeated as facts or used as a success gate. The reattached G0-G7 request is byte-identical to the previously executed plan. Original stages/negative gates are preserved; the review extension is separately registered in prereg_review1.md.

1. **One contribution and presentation.** The body now concerns one retrospective counter-log task: measurement contract, proxy-guided retrieval, detection and recovery cost. State-only/fault views, full learning/calibration, M300 and the unproved general adaptivity detail are in appendices inside the same PDF. The user's earlier instruction to retain one paper takes priority over the suggestion to create a separate supplemental paper. No result is silently deleted.

2. **Cap sensitivity and prevalence.** The predeclared sweep uses Q90/Q95/Q99/Q99.5/Q99.9, the exact original fit-only source-balanced cap rule, and all five existing IDF documentary cap scenarios. query_per_record.csv retains all 71,280 policy/cap/budget cases. The body reports both observed prevalence and the detection curves. Q90 makes 114/120 Pixhawk and 196/209 CMU records reference violators, versus13/111 SpeedyBee records. These are stress labels, not rare known hardware faults. Even Q99.9 need not yield rare record-level exceedance. The 2000W case has no reference violators and undefined sensitivity; it is not evidence of physical validity.

3. **Fixed proxy controls and oracle.** Fixed peak centers and fixed peak brackets use no current energy argument; both reuse the same excluded-source proxies and budget. The forbidden-label oracle isolates a violating adjacent pair plus endpoints, so a finest-reference violation is reachable with at most four anchors. Original Q90 uniform/adaptive outcomes match all3,520 stored rows. The adaptive gain remains {r['primary_gain_points']:.2f}points, but its gain difference versus fixed brackets is {r['adaptive_vs_fixed_brackets_points']:.2f}points with an interval spanning zero. Thus the revised paper does not establish distinct added value from adaptivity. It does not choose the new fixed method as a replacement primary endpoint after outcomes. Adaptive retrieval captures {100*op['fraction_of_oracle_opportunity']:.1f}% of the reachable oracle opportunity under this reference contract.

4. **Grouping, weighting and CI.** The main effect/interval are kept as nominal conditional summaries of the originally defined source groups. Equal-source/equal-record and leave-platform/source-out diagnostics are explicit. Known metadata identify four Pixhawk dates,18 CMU dates and no SpeedyBee dates; dates/campaigns are not invented to manufacture a homogeneous cluster definition. Collapsing the unknown SpeedyBee campaign would leave one block and unidentifiable platform variance, so no false campaign-level CI is produced. Welch and wild-t checks do not cure shared fits or prior exposure. The adverse skew simulation is discussed prominently as a possible undercoverage mechanism, not an estimate that the actual primary CI has exactly0.8coverage.

5. **M300 practical equivalence.** A separate post hoc Holm12 TOST analysis uses the previously stated normalized margin0.01, date-paired effects and t df4. All six residual adapter/gamma contrasts support conditional practical equivalence. None of the six direct contrasts establishes equivalence. Small positive residual differences are therefore described together with supported practical equivalence; the direct model is not forced into that conclusion. Neither test validates the unverified CSV mapping or supplies blind external evidence.

6. **Recovery cost in abstract.** The abstract now states approximately {r['adaptive_volume_cost_percent']:.0f}% equal-platform/source recovery-volume increase under the separate2000W recovery class. The full table gives paired source summaries for every policy. The effect is a detection--recovery trade-off, not measured energy or communications savings.

7. **Reference checks.** Official primary identities and relevant text were rechecked for Cimatti et al.2026, Ramos et al.2026 and the DJI OSDK header. The Ramos Proposition2 continuous-iid Beta law and DJI mV/mA field declarations were read directly. The Cimatti publisher page and formal Section3 were read through the web tool; a separate urllib cache returned transport/challenge content and is not called a full-paper download. Acquisition hashes and read scope are in audit/review1_reference_audit.json. These sources do not validate an undocumented CSV export mapping or arbitrary dependent-source coverage.

**Retained limits.** Everything remains DEV or already-seen conditional sensitivity. Original H1,all1,120 fits,raw231/240 availability,final-batch guard and documented failures remain. No new model fitting, dataset substitution, physical fault claim, AI authorship, journal submission or completed human approval is asserted.
'''
 (ROOT/'REVIEWER_RESPONSE_SIMULATED.md').write_text(response,encoding='utf-8')
 report=f'''# Bản chính sau phản biện giả lập

| Phần | Kết quả và giới hạn |
|---|---|
| G0-G2 gốc | Đã thực hiện; tệp mới trùng byte với kế hoạch cũ. Giữ tái lập, đăng ký và rà tiền lệ; không giả lập lại như thí nghiệm mới. |
| G3 | Không có cận thích nghi tổng quát; giữ kết quả hữu hạn trong phụ lục cùng PDF. |
| G4 mở rộng | Đã chạy71.280 trường hợp cap/chính sách/ngân sách, hai baseline cố định và oracle; tái lập đúng3.520 dòng gốc. |
| Đóng góp chính | Hợp đồng phép đo và truy vấn có proxy cho kiểm toán hồ sơ; chưa chứng minh lợi ích riêng của thích nghi so với baseline fixed brackets. |
| Giá phải trả | Giảm bỏ sót gốc{r['primary_gain_points']:.1f}điểm, đồng thời thể tích phục hồi tăng khoảng{r['adaptive_volume_cost_percent']:.0f}%; đã đưa vào abstract. |
| G5-G6 | Giữ1.120fit, kiểm tra layer và hiệu chỉnh/độ phủ âm; bổ sung trọng số, nhóm nguồn, Welch/wild-t và TOST sau phản biện. |
| M300 | Residual tương đương thực tế theo biên0.01 trong replay có điều kiện; direct chưa chứng minh tương đương. Mappingđơnvị và testmù vẫn chưa hợp lệ. |
| Tài liệu | Đã kiểm lại ba nguồn bị hỏi bằng trang/header/PDF gốc; lưu hash và phạm vi đọc. |
| Bản thảo | Một thân bài tập trung; chi tiết nằm trong phụ lục cùng PDF, không tách paper. |

Nơi nộp: đề xuất NCA trước đây vẫn chỉ có điều kiện; trọng tâm hiện là đo lường/kiểm toán và mức phù hợp ML cần tác giả cân nhắc lại. Chưa chọn nơi nộp mới hoặc gửi bài. Không lặp lại xác suất accept chủ quan như một sự thật.

Không thêm dataset, đổi H1, huấn luyện mới hay chạy thêm testmù. Bản giao chỉ được coi là kiểm chứng xong khi QA release ghi PASSED_INTEGRITY_AND_REPRODUCTION.
'''
 (ROOT/'BAO_CAO_PAPER_CHINH.md').write_text(report,encoding='utf-8')
 checkrows=[('G0','done','evidence/v3/results/g0/G0_gate.json','Exacthistoricaltablesandlimitationsretained'),('G1','done','evidence/v3/results/g1/G1_gate.json','boundedreviewandthreeprimaryreferencesrechecked'),('G2','done','audit/review1_registration.json','newextensionlockedafterdisclosedexposure'),('G3','downgraded','evidence/v3/results/g3/G3_gate.json','nogeneraladaptivitytheoremclaimed'),('G4','done','results/review1/experiment_gate.json','fixedbaselines,oracle,capsandcost'),('G5','done','evidence/v3/results/g5/G5_gate.json','all1120retainednonewfit'),('G6','done','results/review1/statistics_gate.json','weight/group/TOSTsensitivity,noinventedindependence'),('G7','working_manuscript','main.tex','singlefocusedpaper,humanreviewpending')]
 write(out/'requirements_reconciliation.json',{'original_request_sha256':'1a1124be103a72e67dc5c028478d3916b2444203af47d9b6094db30944915d39','gates':[dict(zip(['gate','status','evidence','scope'],row)) for row in checkrows],'all_scientific_extensions_executed':True,'one_PDF':True,'new_blind_result':False})
 log('REVIEW1 RESPONSE/REQUIREMENTS WRITTEN from retained results; earlierstagegatespreserved,allsevenreviewpointsaddressedwithlimits. No external message or submission.')
if __name__=='__main__':main()
