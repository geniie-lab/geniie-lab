import ir_measures
from geniie_lab.dataclasses.measure import Qrels, Run

class MeasureService:

    def calc(self, measures, input_qrels: Qrels, input_run: Run) -> dict:
        """
        Calculate the measures for the given qrels and run.

        Args:
            measures (list): List of measures to calculate. e.g., [ir_measures.nDCG@10, ir_measures.P@10]
            qrels (Qrels): Qrels object containing relevance judgments.
            run (Run): Run object containing the retrieved documents.

        Returns:
            dict: Dictionary of calculated measures.
        """

        qrels = []
        for qrel in input_qrels:
            qrels.append(ir_measures.Qrel(qrel.query_id, qrel.doc_id, qrel.relevance))

        run = []
        for run_item in input_run:
            run.append(ir_measures.ScoredDoc(run_item.query_id, run_item.doc_id, run_item.score))

        results = {}
        agg = ir_measures.calc_aggregate(measures, qrels, run)
        for measure, value in agg.items():
            results[f"{measure}"] = value
        
        return results
