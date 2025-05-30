import os
import shutil
import argparse
import sys
from pathlib import Path

class FAISSDatabaseCleaner:
    def __init__(self, database_path="faiss_logo_database"):
        self.database_path = database_path
    
    def check_database_exists(self):
        """Check if database exists"""
        return os.path.exists(self.database_path)
    
    def get_database_info(self):
        """Get information about the database"""
        if not self.check_database_exists():
            return {"exists": False}
        
        total_size = 0
        files = []
        
        for root, dirs, filenames in os.walk(self.database_path):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                size = os.path.getsize(filepath)
                total_size += size
                files.append({
                    "name": filename,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2)
                })
        
        return {
            "exists": True,
            "path": self.database_path,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": files,
            "file_count": len(files)
        }
    
    def clear_database(self, force=False):
        """Clear the entire database"""
        if not self.check_database_exists():
            print(f"❌ Database not found at: {self.database_path}")
            return False
        
        if not force:
            info = self.get_database_info()
            print(f"📊 Database Info:")
            print(f"   Path: {info['path']}")
            print(f"   Size: {info['total_size_mb']} MB")
            print(f"   Files: {info['file_count']}")
            
            confirm = input("\n⚠️  Are you sure you want to delete the database? (yes/no): ")
            if confirm.lower() not in ['yes', 'y']:
                print("❌ Operation cancelled")
                return False
        
        try:
            shutil.rmtree(self.database_path)
            print(f"✅ Database successfully deleted: {self.database_path}")
            return True
        except Exception as e:
            print(f"❌ Error deleting database: {e}")
            return False
    
    def clear_specific_files(self, file_patterns=None):
        """Clear specific files from database"""
        if not self.check_database_exists():
            print(f"❌ Database not found at: {self.database_path}")
            return False
        
        if file_patterns is None:
            file_patterns = ["*.bin", "*.pkl"]
        
        deleted_files = []
        
        for pattern in file_patterns:
            files = list(Path(self.database_path).glob(pattern))
            for file_path in files:
                try:
                    os.remove(file_path)
                    deleted_files.append(str(file_path))
                    print(f"✅ Deleted: {file_path}")
                except Exception as e:
                    print(f"❌ Error deleting {file_path}: {e}")
        
        if deleted_files:
            print(f"✅ Successfully deleted {len(deleted_files)} files")
        else:
            print("⚠️  No files found matching the patterns")
        
        return len(deleted_files) > 0
    
    def backup_database(self, backup_path=None):
        """Create a backup of the database before clearing"""
        if not self.check_database_exists():
            print(f"❌ Database not found at: {self.database_path}")
            return False
        
        if backup_path is None:
            backup_path = f"{self.database_path}_backup"
        
        try:
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            
            shutil.copytree(self.database_path, backup_path)
            print(f"✅ Database backed up to: {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="FAISS Database Cleaner")
    parser.add_argument("--path", default="faiss_logo_database", 
                       help="Path to FAISS database directory")
    parser.add_argument("--action", choices=["info", "clear", "backup", "clear-files"], 
                       default="info", help="Action to perform")
    parser.add_argument("--force", action="store_true", 
                       help="Force deletion without confirmation")
    parser.add_argument("--backup-path", help="Custom backup path")
    
    args = parser.parse_args()
    
    cleaner = FAISSDatabaseCleaner(args.path)
    
    print(f"🔍 FAISS Database Cleaner")
    print(f"📁 Database Path: {args.path}")
    print("-" * 50)
    
    if args.action == "info":
        info = cleaner.get_database_info()
        if info["exists"]:
            print(f"📊 Database Information:")
            print(f"   ✅ Status: Exists")
            print(f"   📁 Path: {info['path']}")
            print(f"   📏 Size: {info['total_size_mb']} MB ({info['total_size_bytes']} bytes)")
            print(f"   📄 Files: {info['file_count']}")
            print(f"\n📋 File Details:")
            for file_info in info['files']:
                print(f"   - {file_info['name']}: {file_info['size_mb']} MB")
        else:
            print(f"❌ Database does not exist at: {args.path}")
    
    elif args.action == "clear":
        success = cleaner.clear_database(force=args.force)
        if success:
            print("🎉 Database cleared successfully!")
        else:
            print("❌ Failed to clear database")
            sys.exit(1)
    
    elif args.action == "backup":
        success = cleaner.backup_database(args.backup_path)
        if success:
            print("🎉 Database backed up successfully!")
        else:
            print("❌ Failed to backup database")
            sys.exit(1)
    
    elif args.action == "clear-files":
        success = cleaner.clear_specific_files()
        if success:
            print("🎉 Database files cleared successfully!")
        else:
            print("❌ Failed to clear database files")
            sys.exit(1)

if __name__ == "__main__":
    main()



##python clean_database.py --action clear
