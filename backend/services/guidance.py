import os
from typing import List, Dict, Any
from google import genai
from google.genai.errors import APIError

class RepairGuidanceService:
    def __init__(self):
        # The client will automatically use GEMINI_API_KEY from environment variables.
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client()
            except Exception as e:
                print(f"Error initializing Google GenAI client: {e}")

    def generate_guide(self, damages: List[Dict[str, Any]]) -> str:
        """
        Generates a repair guide using Gemini API. If the API is unavailable,
        falls back to a rule-based generator.
        """
        if not damages:
            return "### Inspection Summary\nNo exterior damages were detected. The vehicle's overbody is in excellent condition."

        # Construct textual summary of damages for the prompt
        damage_descriptions = []
        for d in damages:
            cls = d["class"].replace("_", " ").title()
            sev = d["severity"]
            metrics = d["metrics"]
            metric_desc = []
            if "length_cm" in metrics:
                metric_desc.append(f"length: {metrics['length_cm']}cm")
            if "depth_cm" in metrics:
                metric_desc.append(f"depth: {metrics['depth_cm']}cm")
            if "cm2_area" in metrics:
                metric_desc.append(f"affected area: {metrics['cm2_area']}cm²")
            
            desc = f"- **{cls}** ({sev} severity) -> " + ", ".join(metric_desc)
            damage_descriptions.append(desc)

        damages_text = "\n".join(damage_descriptions)
        
        prompt = f"""
You are an expert automotive damage assessor and repair adviser.
Below is a list of detected damages on a vehicle's exterior surface:
{damages_text}

Please generate a professional, structured vehicle repair report and repair guide in Markdown.
The report should include the following sections:
1. **Overall Condition Assessment**: Summarize the general state of the vehicle's overbody.
2. **Step-by-Step Action Plan**: Provide detailed, specific repair steps for each detected issue (e.g., sanding, body filler, primer, paint matching, part replacement).
3. **Recommended Tools & Part Categories**: List specific tools (e.g., dual-action sander, fiberglass mesh, micro-applicators) and generic replacement part suggestions (e.g., passenger headlight assembly).
4. **Estimated Cost Breakdown**: Provide a cost comparison table comparing estimated **DIY Repair Costs** vs. **Professional Shop Repair Costs** for these specific damages.

Be concise, realistic, and make sure the markdown formatting is clean and professional. Do not write generic introductory text. Start directly with the report headers.
"""
        
        if self.client:
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt
                )
                return response.text
            except APIError as e:
                print(f"Gemini API Error: {e}. Falling back to rule-based guidance.")
            except Exception as e:
                print(f"Unexpected error calling Gemini API: {e}. Falling back.")

        return self._generate_fallback_guide(damages)

    def _generate_fallback_guide(self, damages: List[Dict[str, Any]]) -> str:
        """
        Generates a detailed, rule-based repair guide in case the Gemini API is offline.
        """
        severity_counts = {"Mild": 0, "Moderate": 0, "Severe": 0}
        for d in damages:
            severity_counts[d["severity"]] += 1

        overall_status = "Good"
        if severity_counts["Severe"] > 0:
            overall_status = "Poor (Urgent Action Required)"
        elif severity_counts["Moderate"] > 0:
            overall_status = "Fair (Repairs Recommended)"

        report = f"## Vehicle Overbody Repair Report (Local Offline Output)\n\n"
        report += f"### 1. Overall Condition Assessment\n"
        report += f"- **Overall Severity Rating:** {overall_status}\n"
        report += f"- **Total Issues Detected:** {len(damages)}\n"
        report += f"- **Breakdown:** {severity_counts['Mild']} Mild, {severity_counts['Moderate']} Moderate, {severity_counts['Severe']} Severe\n\n"

        report += "### 2. Action Plan per Finding\n"
        for i, d in enumerate(damages, 1):
            cls = d["class"].replace("_", " ").title()
            sev = d["severity"]
            metrics = d["metrics"]
            
            report += f"#### Finding #{i}: {cls} ({sev} Severity)\n"
            
            if d["class"] == "scratch":
                length = metrics.get("length_cm", 5)
                if sev == "Mild":
                    report += f"- **Steps:** Use a scratch repair kit with polishing compound. Clean the area with isopropyl alcohol, apply compound with a microfiber cloth in circular motions, and apply touch-up clear coat if scratch is deep.\n"
                else:
                    report += f"- **Steps:** Deep scratch ({length}cm) requires wet-sanding with 2000-grit sandpaper, applying filler primer, spraying color-matched paint, and finishing with a high-gloss 2K clear coat.\n"
                    
            elif d["class"] == "dent":
                depth = metrics.get("depth_cm", 0.5)
                if sev == "Mild":
                    report += f"- **Steps:** Mild dent ({depth}cm depth) can be pulled using a suction cup dent puller or hot glue puller kit. Clean panel, attach glue tab, let cool, and gently pop back to flush.\n"
                else:
                    report += f"- **Steps:** Large dent ({depth}cm depth) requires Paintless Dent Repair (PDR) rods from the inner side of the panel, or drilling, pulling with slide hammer, applying body filler (Bondo), block-sanding, and repainting.\n"
                    
            elif d["class"] == "rust":
                area = metrics.get("spread_area_cm2", 10)
                report += f"- **Steps:** Affected area is {area}cm². Must grind down to bare metal using 80-grit sandpaper. Apply rust converter to stop chemical corrosion. Apply body filler to build surface level, followed by sanding, priming, and color paint.\n"
                
            elif d["class"] == "crack":
                report += f"- **Steps:** Clean the crack surface. For plastic bumpers, use a plastic welding kit with reinforcing steel wire mesh. For metal panels, welding or panel replacement is recommended.\n"
                
            elif d["class"] == "glass_shatter":
                report += f"- **Steps:** Severe damage to glass. Immediate replacement of the window or windshield is required for safety. Do not attempt temporary glue repair for shatters.\n"
                
            elif d["class"] == "broken_lamp":
                report += f"- **Steps:** Replacement of the outer lens or entire lamp housing. Unclip wiring harness, unscrew mounting brackets, slide in new assembly, and reconnect wiring.\n"
            
            report += "\n"

        report += "### 3. Recommended Tools & Material Categories\n"
        tools = set()
        parts = set()
        for d in damages:
            if d["class"] in ["scratch", "dent", "rust"]:
                tools.update(["Sandpaper (80-2000 grit)", "Body Filler (Bondo)", "Microfiber Towels", "Polishing Compound"])
            if d["class"] == "dent":
                tools.update(["Glue Dent Puller Kit", "Slide Hammer"])
            if d["class"] == "rust":
                tools.update(["Wire Brush / Angle Grinder", "Rust Converter Spray"])
            if d["class"] == "broken_lamp":
                parts.add("Replacement Headlight/Tail-light Housing")
            if d["class"] == "glass_shatter":
                parts.add("OEM Replacement Windshield/Window Glass")
                
        report += "- **Required Tools:** " + ", ".join(tools) + "\n"
        if parts:
            report += "- **Replacement Parts:** " + ", ".join(parts) + "\n"
        else:
            report += "- **Replacement Parts:** No major replacements required. Panel restoration is viable.\n"
        report += "\n"

        report += "### 4. Estimated Cost Range (DIY vs. Shop)\n"
        report += "| Damage Item | Est. DIY Cost | Est. Professional Cost |\n"
        report += "| :--- | :--- | :--- |\n"
        for d in damages:
            cls = d["class"].replace("_", " ").title()
            if d["class"] == "scratch":
                diy, shop = ("$25 - $60", "$200 - $500") if d["severity"] == "Mild" else ("$80 - $150", "$600 - $1200")
            elif d["class"] == "dent":
                diy, shop = ("$30 - $70", "$150 - $400") if d["severity"] == "Mild" else ("$120 - $250", "$800 - $2000")
            elif d["class"] == "rust":
                diy, shop = ("$50 - $100", "$400 - $1000")
            elif d["class"] == "crack":
                diy, shop = ("$40 - $90", "$300 - $700")
            elif d["class"] == "glass_shatter":
                diy, shop = ("N/A (Safety Risk)", "$250 - $600")
            else: # broken_lamp
                diy, shop = ("$50 - $150", "$200 - $450")
            report += f"| {cls} | {diy} | {shop} |\n"
        
        return report
