package com.au.example;
import java.io.IOException;  
import java.util.StringTokenizer;  
  
import org.apache.hadoop.conf.Configuration;  
import org.apache.hadoop.fs.Path;  
import org.apache.hadoop.io.IntWritable;  
import org.apache.hadoop.io.LongWritable;  
import org.apache.hadoop.io.Text;  
import org.apache.hadoop.mapreduce.Job;  
import org.apache.hadoop.mapreduce.Mapper;  
import org.apache.hadoop.mapreduce.Reducer;  
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;  
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;  
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;  
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;  
import org.apache.hadoop.util.GenericOptionsParser;
public class KeyWordCount{
    public static class Map extends Mapper<LongWritable,Text,Text,IntWritable>{
        public final static IntWritable one=new IntWritable(1);
        @override
        public void map(LongWritable key,Text value,Context context) throws Exception{
            String line=value.toString();
            StringTokenizer tokenizerText=new StringTokenizer(line,'\n');
            while(tokenizerText.hasMoreElements()){/*这里tokenizerText是对整个String类按行做的迭代器，调用它的nextToken方法生成对每一行的迭代器对象tokenizerLine*/
                StringTokenizer tokenizerLine=new StringTokenizer(tokenizerText.nextToken());
                String c1=tokenizerLine.nextToken();
                String c2=tokenizerLine.nextToken();
                Text new_text=new Text(c2);
                context.write(new_text,one);
            }
            
        }
    }
    public static class Reduce extends Reducer<Text,IntWritable,Text,IntWritable>{
        public IntWritable result=new IntWritable();
        @override
        public void reduce(Text key,Iterable<IntWritable> values,Context context) throws Exception{
            int count=0;
            for(IntWritable value:values){
                count+=value.get();
                
            }
            result.set(count);
            context.write(key,reult);
            
        }
    }
    public static void main(String[] args) throws Exception{
        Configuration conf=new Configuration();
        if(args.length!=2){
            System.err.println('There must be an input and a output');
            System.exit(2);
        }
        Job job=Job.getInstance(conf,'key word counts');
        job.setJarByClass(KeyWordCount.class);
        job.setMapperClass(KeyWordCount.Map.class);
        job.setCombinerClass(KeyWordCount.Reduce.class);
        job.setReducerClass(KeyWordCount.Reduce.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job,new Path(args[0]));
        FileOutputFormat.setOutputPath(job,new Path(args[1]));
        System.exit(job.waitForCompletion(true)?0:1);
        
    }
}
